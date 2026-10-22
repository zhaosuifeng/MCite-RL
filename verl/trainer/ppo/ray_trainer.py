# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import json
import ast
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_retrieval_refusal_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import re
from mcite_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert (config.data.train_batch_size * config.actor_rollout_ref.rollout.get("n_agent", 1)
                    >= config.actor_rollout_ref.actor.ppo_mini_batch_size)
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            # batch_size=len(self.val_dataset),
            batch_size=505,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        # assert len(
        #     self.val_dataloader
        # ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        print(f'Val dataset size: {len(self.val_dataset)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
        
        # Note: Validation log is written in real-time during _validate(), 
        # so no need to save it here

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # Agent config preparation
        # 获取当前checkpoint目录（用于保存模型输入图像）
        current_checkpoint_dir = self.config.trainer.default_local_dir
        
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_prompt_length=99999,
            num_gpus=self.config.trainer.n_gpus_per_node,
            search_url = self.config.retriever.url,
            checkpoint_dir=current_checkpoint_dir,  # 传递checkpoint目录用于保存模型输入图像
        )

        generation_manager = LLMGenerationManager(
            processor=self.processor,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = batch.repeat_deepcopy(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                if 'multi_modal_inputs' in gen_batch.non_tensor_batch.keys():
                    gen_batch = gen_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = gen_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                with _timer('step', timing_raw):
                    # first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
                    first_input_ids = gen_batch.batch['input_ids'].clone().long()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                        )

                    # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                    for key in final_gen_batch_output.batch.keys():
                        final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()
                    
                    batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                    batch = batch.union(final_gen_batch_output)

                    # special design for RAG dataset for reward func
                    # file_name and reference_page are stored in extra_info dict, not as separate fields
                    # Also pass citation_converted from meta_info to extra_info for reward calculation
                    extra_info_list = []
                    citation_converted_list = batch.meta_info.get('citation_converted', [None] * len(batch.non_tensor_batch['extra_info']))
                    citation_original_list = batch.meta_info.get('citation_original', [None] * len(batch.non_tensor_batch['extra_info']))
                    
                    for idx, extra_info in enumerate(batch.non_tensor_batch['extra_info']):
                        if isinstance(extra_info, dict):
                            # Extract from extra_info dict (standard format)
                            file_name = extra_info.get('file_name', None)
                            reference_page = extra_info.get('reference_page', None)
                            question = extra_info.get('question', None)
                            answer = extra_info.get('answer', None)
                            bounding_box = extra_info.get('bounding_box', None)
                        else:
                            # Fallback: try to get from batch.non_tensor_batch if they exist as separate fields (legacy format)
                            file_name = batch.non_tensor_batch['file_name'][idx] if 'file_name' in batch.non_tensor_batch and idx < len(batch.non_tensor_batch['file_name']) else None
                            reference_page = batch.non_tensor_batch['reference_page'][idx] if 'reference_page' in batch.non_tensor_batch and idx < len(batch.non_tensor_batch['reference_page']) else None
                            question = None
                            answer = None
                            bounding_box = None
                        
                        # Get citation_converted and citation_original from meta_info
                        citation_converted = citation_converted_list[idx] if idx < len(citation_converted_list) else None
                        citation_original = citation_original_list[idx] if idx < len(citation_original_list) else None
                        
                        extra_info_dict = dict(
                            file_name=file_name, 
                            reference_page=reference_page, 
                            question=question, 
                            answer=answer
                        )
                        
                        # Add bounding_box if exists
                        if bounding_box is not None:
                            extra_info_dict['bounding_box'] = bounding_box
                        
                        # Add citation_converted and citation_original from meta_info (if available)
                        if citation_converted is not None:
                            extra_info_dict['citation_converted'] = citation_converted
                        if citation_original is not None:
                            extra_info_dict['citation_original'] = citation_original
                        
                        extra_info_list.append(extra_info_dict)
                    batch.non_tensor_batch['extra_info'] = np.array(extra_info_list)
                    for idx in range(len(batch.non_tensor_batch['raw_prompt_ids'])):
                        batch.non_tensor_batch['raw_prompt_ids'][idx] = [int(item) for item in batch.non_tensor_batch['raw_prompt_ids'][idx]]
                        

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        with torch.no_grad():
                            print('========= compute log prob =========')
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        print('===============ref log prob==================')
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        print('========= compute advantage =========')
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n_agent)
                        print('========= compute advantage done =========')

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            batch, metrics = self._create_loss_mask(batch, metrics)                            
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                            
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)


                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_retrieval_refusal_metrics(batch=batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1
    # mask user and last noisy
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        # Initialize state mask
        state_mask = torch.ones_like(response_mask)
        
        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) for resp in batch.batch['responses']]
    
        for i, response in enumerate(responses):
            # Find all pairs of start and end marker positions
            # start_marker = self.config.algorithm.state_masking.start_state_marker
            # end_marker = self.config.algorithm.state_masking.end_state_marker   
            start_marker = "\n<|im_start|>user\n"
            end_marker = "<|im_end|>\n<|im_start|>assistant\n"
            
            # Get all start and end positions
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            
            # Convert character positions to token positions
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[:start]
                state_section = response[start:end]
                
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)
                
                start_token_pos = len(start_tokens)
                end_token_pos = start_token_pos + len(state_tokens)
                
                state_mask[i, start_token_pos:end_token_pos] = 0
        

        ######## find last \n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|><|im_end|>
        for i, response in enumerate(responses): 
            start_marker = "\n<|im_start|>user\n"
            end_marker = "<|im_end|>"
            # Get all start and end positions
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            # Convert character positions to token positions
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[:start]
                state_section = response[start:end]
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)
                start_token_pos = len(start_tokens)
                end_token_pos = start_token_pos + len(state_tokens)
                state_mask[i, start_token_pos:end_token_pos] = 0
            
        loss_mask = state_mask * response_mask
        batch.batch['loss_mask'] = loss_mask
        
        # # # Debug print
        # print("\nRaw batch[0] (before masking):\n", self.tokenizer.decode(batch.batch['responses'][0]))
        # response_ids = batch.batch['responses'][0]
        # unmasked_ids = response_ids[loss_mask[0] == 0]
        # print("\nMasked batch[0] (after masking):\n", self.tokenizer.decode(unmasked_ids))
        
        # masked_ids = response_ids[loss_mask[0] == 1]
        # print("\nUnmasked batch[0] (masked parts):\n", self.tokenizer.decode(masked_ids))
        
        # masked_ids = response_ids[response_mask[0] == 1]
        # print("\nresponse_mask[0] == 1:\n", self.tokenizer.decode(masked_ids))
        
        # masked_ids = response_ids[response_mask[0] == 0]
        # print("\nresponse_mask[0] == 0:\n", self.tokenizer.decode(masked_ids))

        # metrics.update({
        #     'state_tokens/total': loss_mask.sum().item(),
        #     'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        # })
        
        return batch, metrics

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        import json
        from verl.utils.reward_score.mcite import (
            get_answer_from_predict_str,
            remove_text_between_tags,
            compute_comprehensive_evaluation,
        )
        
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        
        # Collect validation results for logging
        validation_logs = []
        
        # Collect comprehensive evaluation stats
        all_citation_ious = []  # only for valid cases
        all_bbox_rewards = []  # 累积bbox_reward
        search_correct_total = 0
        # Refusal stats (validation)
        refusal_total = 0
        refusal_on_search_success = 0
        refusal_on_search_failure = 0
        search_success_total = 0
        search_failure_total = 0
        answer_exact_correct_total = 0
        answer_containment_correct_total = 0
        answer_exact_correct_on_search_success = 0
        answer_containment_correct_on_search_success = 0
        has_citation_total = 0
        has_answer_total = 0
        comprehensive_valid_count = 0
        total_eval_samples = 0
        all_relevance_scores = []
        
        # Setup validation log file path (real-time logging)
        # Use current global_steps for the log file path
        step_dir = os.path.join(self.config.trainer.default_local_dir, f'global_step_{self.global_steps}')
        validation_log_path = os.path.join(step_dir, 'validation_log.jsonl')
        
        # 保存推理结果文件（格式与inference_wiki_visa.py一致，可用evaluate_inference.py评测）
        inference_results_path = os.path.join(step_dir, 'validation_inference_results.jsonl')
        
        # Create directory if it doesn't exist
        os.makedirs(step_dir, exist_ok=True)
        
        # Clear or create the log file at the start of validation
        with open(validation_log_path, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        print(f'Validation log will be saved in real-time to: {validation_log_path}')
        
        # Clear inference results file
        with open(inference_results_path, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        print(f'Validation inference results will be saved to: {inference_results_path}')

        # 获取当前checkpoint目录（用于保存模型输入图像）
        current_checkpoint_dir = step_dir  # 使用validation的step_dir
        
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_prompt_length=99999,
            num_gpus=self.config.trainer.n_gpus_per_node,
            search_url = self.config.retriever.url,
            checkpoint_dir=current_checkpoint_dir,  # 传递checkpoint目录用于保存模型输入图像
        )

        generation_manager = LLMGenerationManager(
            processor=self.processor,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True
        )

        
        print(f'Starting validation with {len(self.val_dataloader)} batches')
        
        # Set checkpoint directory for val_reward_fn if it's RMManager in eval mode
        if self.val_reward_fn is not None and hasattr(self.val_reward_fn, 'eval_mode') and self.val_reward_fn.eval_mode:
            self.val_reward_fn.checkpoint_dir = step_dir
            print(f'[Checkpoint] Set checkpoint_dir for val_reward_fn: {step_dir}')
        else:
            print(f'[Checkpoint] Warning: val_reward_fn is None or not in eval mode. val_reward_fn={self.val_reward_fn}, has_eval_mode={hasattr(self.val_reward_fn, "eval_mode") if self.val_reward_fn is not None else False}, eval_mode={getattr(self.val_reward_fn, "eval_mode", None) if self.val_reward_fn is not None else None}')
        
        for batch_idx, test_data in enumerate(self.val_dataloader):
            print(f'Processing validation batch {batch_idx + 1}/{len(self.val_dataloader)}')
            timing_raw = {}
            test_batch = DataProto.from_single_dict(test_data)
            print(f'Validation batch size: {test_batch.batch["input_ids"].shape[0]}')

            # repeat test batch
            test_batch = test_batch.repeat_deepcopy(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            # if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
            #     return {}

            # Store original inputs
            # input_ids = test_batch.batch['input_ids']
            # input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            # sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'].clone().long()
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        initial_input_ids=first_input_ids,
                    )
                
                test_batch = test_batch.union(final_gen_batch_output)
                
                # 收集citation IoU统计信息
                if hasattr(generation_manager, 'citation_ious') and len(generation_manager.citation_ious) > 0:
                    all_citation_ious.extend(generation_manager.citation_ious)
                
                for key in test_batch.batch.keys():
                    test_batch.batch[key] = test_batch.batch[key].long()
                
                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                # try:
                print(f'[Validation] Calling val_reward_fn: type={type(self.val_reward_fn)}, eval_mode={getattr(self.val_reward_fn, "eval_mode", "N/A")}, checkpoint_dir={getattr(self.val_reward_fn, "checkpoint_dir", "N/A")}')
                reward_tensor = self.val_reward_fn(test_batch)
                # except:
                    # print(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                
                # Collect validation results for logging
                for i in range(len(test_batch)):
                    try:
                        data_item = test_batch[i]
                        
                        # Get prompt IDs - try 'prompts' first, fallback to 'input_ids'
                        if 'prompts' in data_item.batch:
                            prompt_ids = data_item.batch['prompts']
                            prompt_length = prompt_ids.shape[-1]
                            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                        else:
                            # Fallback to input_ids if prompts not available
                            input_ids = data_item.batch['input_ids']
                            attention_mask = data_item.batch['attention_mask']
                            # Find the prompt length (before response)
                            prompt_length = attention_mask.sum().item() - data_item.batch['responses'].shape[-1]
                            valid_prompt_ids = input_ids[:prompt_length]
                        
                        response_ids = data_item.batch['responses']
                        attention_mask = data_item.batch['attention_mask']
                        valid_response_length = attention_mask.sum().item() - len(valid_prompt_ids)
                        valid_response_ids = response_ids[:valid_response_length] if valid_response_length > 0 else response_ids
                        
                        # Decode prompt and response
                        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                        
                        # Extract question from prompt or extra_info
                        extra_info = data_item.non_tensor_batch.get('extra_info', None)
                        if extra_info is not None and isinstance(extra_info, dict) and 'question' in extra_info:
                            question = extra_info['question']
                        else:
                            # Try to extract question from prompt
                            question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', prompt_str, re.IGNORECASE)
                            question = question_match.group(1).strip() if question_match else prompt_str[:200] + "..."
                        
                        # Extract predicted answer
                        cleaned_response = remove_text_between_tags(response_str)
                        predicted_answer = get_answer_from_predict_str(cleaned_response)
                        if predicted_answer is None:
                            predicted_answer = ""

                        # Refusal flag for validation metrics (heuristic by default).
                        # To enable LLM judge for refusal in validation, set:
                        #   VAL_REFUSAL_USE_LLM_JUDGE=1
                        # Note: this can be slow/costly because it calls an external API per sample.
                        is_refusal = False
                        try:
                            use_llm_refusal = os.environ.get("VAL_REFUSAL_USE_LLM_JUDGE", "0").lower() in ("1", "true", "yes", "y")
                            if use_llm_refusal and self.val_reward_fn is not None and hasattr(self.val_reward_fn, "_judge_refusal_with_llm"):
                                q_for_refusal = question or ""
                                is_refusal = bool(self.val_reward_fn._judge_refusal_with_llm(q_for_refusal, predicted_answer or ""))
                            else:
                                from verl.workers.reward_manager.rm import is_refusal_answer
                                is_refusal = bool(is_refusal_answer(predicted_answer))
                        except Exception:
                            is_refusal = False
                        
                        # Get ground truth
                        ground_truth = data_item.non_tensor_batch.get('reward_model', {}).get('ground_truth', '')
                        if not ground_truth and extra_info is not None and isinstance(extra_info, dict):
                            ground_truth = extra_info.get('answer', '')
                        
                        # Get score - reward_tensor has shape (batch_size, seq_len), need to sum to get sample-level score
                        if i < reward_tensor.shape[0]:
                            # Sum all token-level rewards for this sample to get the final score
                            score = reward_tensor[i].sum().item()
                        else:
                            score = 0.0
                        # Build extra_info for comprehensive evaluation
                        # Base extra_info from sample
                        base_extra_info = extra_info if isinstance(extra_info, dict) else {}
                        # Retrieval images and crop history per-sample if available
                        retrievaled_images = test_batch.non_tensor_batch.get('retrievaled_images', None)
                        retrievaled_images_i = retrievaled_images[i] if isinstance(retrievaled_images, np.ndarray) and i < len(retrievaled_images) else None
                        crop_history = getattr(generation_manager, 'crop_history', None)
                        if crop_history is None:
                            crop_history = test_batch.non_tensor_batch.get('crop_history', None)
                        crop_history_i = crop_history[i] if isinstance(crop_history, list) and i < len(crop_history) else (crop_history[i] if isinstance(crop_history, np.ndarray) and i < len(crop_history) else None)
                        # Ground truth citation fallback from extra_info['bounding_box']
                        gt_citation = base_extra_info.get('ground_truth_citation') or base_extra_info.get('bounding_box')
                        # Compose eval extra info dict
                        eval_extra_info = dict(base_extra_info)
                        if retrievaled_images_i is not None:
                            eval_extra_info['retrievaled_images'] = retrievaled_images_i
                        if crop_history_i is not None:
                            eval_extra_info['crop_history'] = crop_history_i
                        if gt_citation is not None:
                            eval_extra_info['ground_truth_citation'] = gt_citation

                        # Get relevance score from non_tensor_batch
                        relevance_score = data_item.non_tensor_batch.get('reward_model', {}).get('relevance_score', None)

                        # Run comprehensive evaluation (aligned with evaluate_inference.py)
                        eval_result = compute_comprehensive_evaluation(
                            response_str, ground_truth, eval_extra_info
                        )
                        # -------- bbox_reward 汇总 --------
                        from verl.utils.reward_score.mcite import evaluate_bbox_reward
                        bbox_reward_val = evaluate_bbox_reward(response_str, eval_extra_info)
                        if bbox_reward_val is not None:
                            all_bbox_rewards.append(bbox_reward_val)
                        # -------- END bbox_reward 汇总 --------
                        # Accumulate comprehensive metrics
                        total_eval_samples += 1
                        sc_flag = bool(eval_result.get('search_correct', False))
                        if sc_flag:
                            search_correct_total += 1
                            search_success_total += 1
                            if is_refusal:
                                refusal_on_search_success += 1
                            # Count answer correctness when search is correct
                            if eval_result.get('answer_exact_correct'):
                                answer_exact_correct_on_search_success += 1
                            if eval_result.get('answer_containment_correct'):
                                answer_containment_correct_on_search_success += 1
                        else:
                            search_failure_total += 1
                            if is_refusal:
                                refusal_on_search_failure += 1
                        if is_refusal:
                            refusal_total += 1
                        if eval_result.get('has_answer'):
                            has_answer_total += 1
                        if eval_result.get('has_citation'):
                            has_citation_total += 1
                        if eval_result.get('answer_exact_correct'):
                            answer_exact_correct_total += 1
                        if eval_result.get('answer_containment_correct'):
                            answer_containment_correct_total += 1
                        citation_iou_val = eval_result.get('citation_iou')
                        if eval_result.get('valid_for_evaluation') and citation_iou_val is not None:
                            comprehensive_valid_count += 1
                            all_citation_ious.append(citation_iou_val)
                        
                        if relevance_score is not None:
                            all_relevance_scores.append(relevance_score)

                        log_entry = {
                            'question': question,
                            'predicted_answer': predicted_answer,
                            'ground_truth': ground_truth,
                            'score': score,
                            'relevance_score': relevance_score,
                            'full_response': response_str[:500] + "..." if len(response_str) > 500 else response_str,
                            'search_correct': sc_flag,
                            'is_refusal': bool(is_refusal),
                            'has_answer': bool(eval_result.get('has_answer', False)),
                            'has_citation': bool(eval_result.get('has_citation', False)),
                            'answer_exact_correct': bool(eval_result.get('answer_exact_correct', False)),
                            'answer_containment_correct': bool(eval_result.get('answer_containment_correct', False)),
                            'citation_iou': citation_iou_val if citation_iou_val is not None else None,
                        }
                        validation_logs.append(log_entry)
                        
                        # Real-time write to log file
                        with open(validation_log_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                            f.flush()  # 立即刷新到磁盘
                        
                        # 保存推理结果（格式与inference_wiki_visa.py一致，可用evaluate_inference.py评测）
                        try:
                            # 获取uid
                            uid = data_item.non_tensor_batch.get('uid', f'sample_{i}')
                            if isinstance(uid, np.ndarray):
                                uid = uid.item() if uid.size > 0 else f'sample_{i}'
                            
                            # 构建all_images列表（从retrievaled_images和crop_history）
                            all_images = []
                            if retrievaled_images_i is not None:
                                if isinstance(retrievaled_images_i, list):
                                    for img_path in retrievaled_images_i:
                                        all_images.append({
                                            'type': 'search_result',
                                            'path': img_path,
                                            'image_id': None  # 可以从路径提取
                                        })
                            
                            # 添加crop_result
                            if crop_history_i is not None:
                                if isinstance(crop_history_i, list):
                                    for crop_info in crop_history_i:
                                        if crop_info.get('bbox') and crop_info.get('image_path'):
                                            all_images.append({
                                                'type': 'crop_result',
                                                'path': crop_info.get('image_path'),
                                                'bbox': crop_info.get('bbox')
                                            })
                            
                            # 构建all_responses（目前只有最终响应，训练时可能没有保存每次迭代）
                            # 为了兼容evaluate_inference.py，我们至少保存最终响应
                            all_responses = [{
                                'iteration': 1,
                                'response': response_str,
                                'messages_before': 0
                            }]
                            
                            # 提取citation（使用与evaluate_inference.py完全一致的方法）
                            # 从最终响应中提取，确保与推理评测逻辑一致
                            citation_original = None
                            citation_converted = None
                            
                            # 使用与evaluate_inference.py完全相同的提取逻辑
                            # 方法1: 标准格式 <citation>[x1, y1, x2, y2]</citation>（大小写不敏感，允许换行）
                            citation_match = re.search(r'<citation>\s*\[([^\]]+)\]\s*</citation>', response_str, re.IGNORECASE | re.DOTALL)
                            if citation_match:
                                try:
                                    citation_original = ast.literal_eval(f"[{citation_match.group(1)}]")
                                    if not (isinstance(citation_original, list) and len(citation_original) == 4):
                                        citation_original = None
                                except:
                                    citation_original = None
                            
                            # 方法2: 查找 </answer> 后面的坐标（可能没有citation标签）
                            if citation_original is None:
                                answer_end_match = re.search(r'</answer>', response_str, re.IGNORECASE)
                                if answer_end_match:
                                    after_answer = response_str[answer_end_match.end():]
                                    coord_match = re.search(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', after_answer)
                                    if coord_match:
                                        try:
                                            citation_original = [int(coord_match.group(1)), int(coord_match.group(2)), 
                                                               int(coord_match.group(3)), int(coord_match.group(4))]
                                        except:
                                            citation_original = None
                            
                            # 方法3: 在response末尾查找坐标（通常citation在最后）
                            if citation_original is None:
                                all_coords = re.findall(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', response_str)
                                if all_coords:
                                    try:
                                        # 取最后一个坐标（通常citation在最后）
                                        last_coord = all_coords[-1]
                                        citation_original = [int(last_coord[0]), int(last_coord[1]), 
                                                           int(last_coord[2]), int(last_coord[3])]
                                    except:
                                        citation_original = None
                            
                            # 转换citation坐标到检索图像坐标系
                            if citation_original is not None:
                                try:
                                    from verl.utils.reward_score.mcite import (
                                        build_crop_chain_from_extra_info,
                                        convert_crop_coords_to_search_image
                                    )
                                    crop_chain = build_crop_chain_from_extra_info(
                                        retrievaled_images_i if retrievaled_images_i else [],
                                        crop_history_i if crop_history_i else []
                                    )
                                    
                                    # 调试信息：打印crop_chain信息
                                    if len(crop_chain) == 0:
                                        print(f"[Warning] Empty crop_chain for sample {i} (uid={uid})")
                                        print(f"  retrievaled_images_i: {retrievaled_images_i}")
                                        print(f"  crop_history_i: {crop_history_i}")
                                    
                                    citation_converted = convert_crop_coords_to_search_image(
                                        citation_original, crop_chain
                                    )
                                    
                                    if citation_converted is None:
                                        print(f"[Warning] Failed to convert citation for sample {i} (uid={uid}): conversion returned None")
                                        print(f"  citation_original: {citation_original}")
                                        print(f"  crop_chain length: {len(crop_chain)}")
                                        if len(crop_chain) > 0:
                                            print(f"  crop_chain: {crop_chain}")
                                    else:
                                        # 调试信息：打印转换结果（仅在IoU较低时）
                                        if gt_citation is not None:
                                            try:
                                                from verl.utils.reward_score.mcite import calculate_iou
                                                if isinstance(gt_citation, str):
                                                    import ast
                                                    gt_citation_parsed = ast.literal_eval(gt_citation)
                                                else:
                                                    gt_citation_parsed = gt_citation
                                                if isinstance(gt_citation_parsed, list) and len(gt_citation_parsed) == 4:
                                                    iou = calculate_iou(citation_converted, gt_citation_parsed)
                                                    if iou < 0.3:
                                                        print(f"[Debug] Low IoU for sample {i} (uid={uid}): {iou:.3f}")
                                                        print(f"  citation_original: {citation_original}")
                                                        print(f"  citation_converted: {citation_converted}")
                                                        print(f"  ground_truth_citation: {gt_citation_parsed}")
                                                        print(f"  crop_chain length: {len(crop_chain)}")
                                            except Exception as e:
                                                pass  # 忽略调试信息的错误
                                except Exception as e:
                                    print(f"[Warning] Failed to convert citation for sample {i} (uid={uid}): {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            # 获取reference信息
                            reference_page = base_extra_info.get('reference_page', None)
                            if reference_page is not None and hasattr(reference_page, 'tolist'):
                                try:
                                    reference_page = reference_page.tolist()
                                except:
                                    reference_page = [reference_page] if reference_page is not None else None
                            
                            # 构建推理结果（格式与inference_wiki_visa.py一致）
                            inference_result = {
                                'uid': str(uid),
                                'query': question,
                                'all_responses': all_responses,
                                'all_images': all_images,
                                'iterations': len(all_responses),
                                'completed': predicted_answer is not None and predicted_answer != "",
                                'citation_original': citation_original,
                                'citation_converted': citation_converted,
                                'reference_answer': ground_truth,
                                'ground_truth_citation': gt_citation,
                                'reference_page': reference_page
                            }
                            
                            # 保存到文件
                            with open(inference_results_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(inference_result, ensure_ascii=False) + '\n')
                                f.flush()
                        except Exception as e:
                            print(f"[Warning] Failed to save inference result for sample {i}: {e}")
                            import traceback
                            traceback.print_exc()
                    except Exception as e:
                        print(f"Error collecting validation log for sample {i}: {e}")
                        continue

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        

        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        metric_dict = {}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            # 调试信息：打印每个data_source的统计
            print(f"[Debug] data_source '{data_source}': count={len(rewards)}, mean={np.mean(rewards):.4f}, min={np.min(rewards):.4f}, max={np.max(rewards):.4f}")

        # overall
        overall_rewards = []
        for i in range(reward_tensor.shape[0]):
            overall_rewards.append(reward_tensor[i].item())
        metric_dict[f'val/test_score/overall'] = np.mean(overall_rewards)
        
        # Calculate accuracy (correct rate) - for EM scoring, score == 1.0 means correct
        data_source_accuracy = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            score = reward_tensor[i].item()
            is_correct = 1.0 if score >= 1.0 else 0.0  # For EM, score == 1.0 means correct
            
            if data_source not in data_source_accuracy:
                data_source_accuracy[data_source] = {'correct': 0, 'total': 0}
            data_source_accuracy[data_source]['total'] += 1
            if is_correct > 0:
                data_source_accuracy[data_source]['correct'] += 1
        
        # Calculate and add accuracy metrics
        for data_source, acc_info in data_source_accuracy.items():
            accuracy = acc_info['correct'] / acc_info['total'] if acc_info['total'] > 0 else 0.0
            metric_dict[f'val/accuracy/{data_source}'] = accuracy
        
        # Overall accuracy
        total_correct = sum([acc_info['correct'] for acc_info in data_source_accuracy.values()])
        total_samples = sum([acc_info['total'] for acc_info in data_source_accuracy.values()])
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        metric_dict[f'val/accuracy/overall'] = overall_accuracy
        
        # ================= Comprehensive evaluation metrics (aligned with evaluate_inference.py) =================
        # Overall search accuracy (full set)
        if total_eval_samples > 0:
            search_accuracy_overall = search_correct_total / total_eval_samples
        else:
            search_accuracy_overall = 0.0
        metric_dict['val/search_accuracy/overall'] = search_accuracy_overall

        # Refusal metrics
        metric_dict['val/refusal_rate/overall'] = (refusal_total / total_eval_samples) if total_eval_samples > 0 else 0.0
        metric_dict['val/refusal_rate_on_retrieval_success'] = (refusal_on_search_success / search_success_total) if search_success_total > 0 else 0.0
        metric_dict['val/refusal_rate_on_retrieval_failure'] = (refusal_on_search_failure / search_failure_total) if search_failure_total > 0 else 0.0
        metric_dict['val/refusal_count/overall'] = float(refusal_total)
        metric_dict['val/refusal_count_on_retrieval_success'] = float(refusal_on_search_success)
        metric_dict['val/refusal_count_on_retrieval_failure'] = float(refusal_on_search_failure)
        
        # Answer accuracies on full dataset (failure to answer counts as wrong)
        if total_eval_samples > 0:
            answer_exact_acc_full = answer_exact_correct_total / total_eval_samples
            answer_containment_acc_full = answer_containment_correct_total / total_eval_samples
        else:
            answer_exact_acc_full = 0.0
            answer_containment_acc_full = 0.0
        # Answer accuracies on full dataset (failure to answer counts as wrong)
        metric_dict['val/answer_exact_accuracy_full_set'] = answer_exact_acc_full
        metric_dict['val/answer_containment_accuracy_full_set'] = answer_containment_acc_full
        # Also add with grouped naming for better wandb visualization
        metric_dict['val/answer_accuracy/exact/full_set'] = answer_exact_acc_full
        metric_dict['val/answer_accuracy/containment/full_set'] = answer_containment_acc_full
        
        # Answer accuracies when search is correct
        if search_success_total > 0:
            answer_exact_acc_on_search_success = answer_exact_correct_on_search_success / search_success_total
            answer_containment_acc_on_search_success = answer_containment_correct_on_search_success / search_success_total
        else:
            answer_exact_acc_on_search_success = 0.0
            answer_containment_acc_on_search_success = 0.0
        metric_dict['val/answer_exact_accuracy_on_search_success'] = answer_exact_acc_on_search_success
        metric_dict['val/answer_containment_accuracy_on_search_success'] = answer_containment_acc_on_search_success
        # Also add with grouped naming for better wandb visualization
        metric_dict['val/answer_accuracy/exact/on_search_success'] = answer_exact_acc_on_search_success
        metric_dict['val/answer_accuracy/containment/on_search_success'] = answer_containment_acc_on_search_success
        
        # Count/ratio of samples with citation and answer presence
        #metric_dict['val/has_answer/count'] = has_answer_total
        has_citation_ratio = has_citation_total / total_eval_samples if total_eval_samples > 0 else 0.0
        #metric_dict['val/has_citation/count'] = has_citation_total
        metric_dict['val/has_citation/ratio'] = has_citation_ratio
        
        # Calculate relevance score statistics
        if len(all_relevance_scores) > 0:
            mean_relevance = np.mean(all_relevance_scores)
            metric_dict['val/relevance_score/mean'] = mean_relevance
            if len(all_relevance_scores) > 1:
                metric_dict['val/relevance_score/std'] = np.std(all_relevance_scores)
                metric_dict['val/relevance_score/median'] = np.median(all_relevance_scores)
        else:
            metric_dict['val/relevance_score/mean'] = 0.0

        # Calculate citation IoU statistics
        if len(all_citation_ious) > 0:
            mean_citation_iou = np.mean(all_citation_ious)
            metric_dict['val/citation_iou/mean'] = mean_citation_iou
            if len(all_citation_ious) > 1:
                std_citation_iou = np.std(all_citation_ious)
                median_citation_iou = np.median(all_citation_ious)
                metric_dict['val/citation_iou/std'] = std_citation_iou
                metric_dict['val/citation_iou/median'] = median_citation_iou
        else:
            mean_citation_iou = None
        # ---- bbox_reward stats ----
        if len(all_bbox_rewards) > 0:
            mean_bbox_reward = np.mean(all_bbox_rewards)
            metric_dict['val/bbox_reward/mean'] = mean_bbox_reward
            if len(all_bbox_rewards) > 1:
                std_bbox_reward = np.std(all_bbox_rewards)
                median_bbox_reward = np.median(all_bbox_rewards)
                metric_dict['val/bbox_reward/std'] = std_bbox_reward
                metric_dict['val/bbox_reward/median'] = median_bbox_reward
        else:
            mean_bbox_reward = None
        # ---- END bbox_reward stats ----
        
        # Print summary with accuracy
        print("\n" + "="*80)
        print("Validation Summary:")
        print(f"Total samples: {total_samples}")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
        print(f"Overall Average Score: {np.mean(overall_rewards):.4f}")
        # Comprehensive summary
        print()
        print(f"Search accuracy (overall): {search_accuracy_overall:.4f} ({search_correct_total}/{total_eval_samples})")
        print("Answer Evaluation (Full Dataset - Failure to Answer is Wrong):")
        print(f"  Exact Match Accuracy: {answer_exact_acc_full * 100:.1f}%")
        print(f"  Containment Accuracy: {answer_containment_acc_full * 100:.1f}%")
        print("Answer Evaluation (When Search is Correct):")
        print(f"  Exact Match Accuracy: {answer_exact_acc_on_search_success * 100:.1f}% ({answer_exact_correct_on_search_success}/{search_success_total})")
        print(f"  Containment Accuracy: {answer_containment_acc_on_search_success * 100:.1f}% ({answer_containment_correct_on_search_success}/{search_success_total})")
        print(f"\nCitation Availability:")
        print(f"  Samples with citation: {has_citation_total}/{total_eval_samples} ({has_citation_ratio * 100:.1f}%)")
        
        # Print relevance stats
        if len(all_relevance_scores) > 0:
            print(f"\nRelevance Score Statistics:")
            print(f"  Mean Relevance: {np.mean(all_relevance_scores):.4f}")
            if len(all_relevance_scores) > 1:
                print(f"  Std Relevance: {np.std(all_relevance_scores):.4f}")
                print(f"  Median Relevance: {np.median(all_relevance_scores):.4f}")

        print("\nPer Data Source:")
        for data_source in sorted(data_source_accuracy.keys()):
            acc_info = data_source_accuracy[data_source]
            accuracy = acc_info['correct'] / acc_info['total'] if acc_info['total'] > 0 else 0.0
            avg_score = np.mean(data_source_reward[data_source]) if data_source in data_source_reward else 0.0
            print(f"  {data_source}: Accuracy={accuracy:.4f} ({acc_info['correct']}/{acc_info['total']}), Avg Score={avg_score:.4f}")
        
        # Print citation IoU statistics
        if mean_citation_iou is not None:
            print(f"\nCitation IoU Statistics:")
            print(f"  Total samples with citation: {len(all_citation_ious)}")
            print(f"  Mean IoU: {mean_citation_iou:.4f}")
            if len(all_citation_ious) > 1:
                print(f"  Std IoU: {std_citation_iou:.4f}")
                print(f"  Median IoU: {median_citation_iou:.4f}")
                # IoU distribution
                iou_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
                print(f"  IoU Distribution:")
                for min_iou, max_iou in iou_ranges:
                    count = sum(1 for iou in all_citation_ious if min_iou <= iou < max_iou)
                    percentage = count / len(all_citation_ious) * 100
                    print(f"    [{min_iou:.1f}, {max_iou:.1f}): {count} ({percentage:.1f}%)")
                high_iou_count = sum(1 for iou in all_citation_ious if iou >= 0.9)
                print(f"    [0.9, 1.0]: {high_iou_count} ({high_iou_count / len(all_citation_ious) * 100:.1f}%)")
        else:
            print(f"\nCitation IoU: No valid citations found")
        # ---- bbox_reward 输出 ----
        if mean_bbox_reward is not None:
            print(f"\nBBox Reward Statistics:")
            print(f"  Total samples: {len(all_bbox_rewards)}")
            print(f"  Mean bbox_reward: {mean_bbox_reward:.4f}")
            if len(all_bbox_rewards) > 1:
                print(f"  Std bbox_reward: {std_bbox_reward:.4f}")
                print(f"  Median bbox_reward: {median_bbox_reward:.4f}")
        else:
            print(f"\nBBox Reward: No valid bbox_reward found")
        # ---- END bbox_reward 输出 ----
        
        print("="*80 + "\n")
        
        # Log file is already written in real-time, just print summary
        if validation_logs:
            print(f'Validation log saved to: {validation_log_path} ({len(validation_logs)} samples)')
        
        return metric_dict

