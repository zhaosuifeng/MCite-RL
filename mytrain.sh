#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
TRAIN_PY="${TRAIN_PY:-python}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export REFUSAL_JUDGE_ENABLED="${REFUSAL_JUDGE_ENABLED:-0}"
export REFUSAL_METRICS_ENABLED="${REFUSAL_METRICS_ENABLED:-0}"

# -----------------------------
# Minimal VRAG PPO launch script
# -----------------------------
# Edit the variables below as needed.

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# -----------------------------
# GPU split (train vs search_api)
# -----------------------------
# Recommended: run search_api on a dedicated GPU (e.g. physical GPU7):
#   CUDA_VISIBLE_DEVICES=7 python search_engine/search_engine_api.py
#
# Override the current single-GPU default when using multiple GPUs:
#   TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash mytrain.sh
#
# NOTE: search_engine uses device_map='cuda:0', so CUDA_VISIBLE_DEVICES remaps GPU IDs.
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$TRAIN_CUDA_VISIBLE_DEVICES}"
ENGINE="vllm"   # Default engine
# Check if the first argument is NOT a key=value pair (does not contain '=')
# If so, treat it as the ENGINE override.
if [ "$#" -gt 0 ] && [[ "$1" != -* && "$1" != *"="* ]]; then
    ENGINE="$1"
    shift
fi

# ----------- User-editable config ----------
model_path="${MODEL_PATH:-models/policy}"
# Single file (default):
train_file="${TRAIN_FILE:-data/wiki-visa/train.parquet}"
# Use wiki_visa-style schema (id/prompt/reward_model/extra_info/...) built from original paper-visa shards,
# so `extra_info.file_name/reference_page` align with retriever IDs like `PMCxxxx_xxxxx`.
##train_file="./data_new/wiki_visa/train_batch_8_from_papervisa_src.parquet"
val_file="${VAL_FILE:-data/wiki-visa/val.parquet}"
#val_file="./data_new/wiki_visa/val_batch_from_papervisa_src_150.parquet"
# Derive n_gpus from CUDA_VISIBLE_DEVICES (keeps trainer/vllm consistent).
# Override manually via N_GPUS if needed.
if [ -n "${N_GPUS:-}" ]; then
  n_gpus="$N_GPUS"
else
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # count comma-separated entries
    n_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
  else
    n_gpus=1
  fi
fi

# total train batch (global). Must be divisible by n_gpus.
desired_train_batch_size="${TRAIN_BATCH_SIZE:-1}"
train_batch_size="$desired_train_batch_size"
if [ $((train_batch_size % n_gpus)) -ne 0 ]; then
  # round UP to the next multiple of n_gpus (keeps per-GPU batch stable)
  train_batch_size=$(( ( (train_batch_size + n_gpus - 1) / n_gpus ) * n_gpus ))
  echo "[WARN] desired_train_batch_size (${desired_train_batch_size}) is not divisible by n_gpus (${n_gpus})."
  echo "[WARN] Adjusting train_batch_size -> ${train_batch_size}."
fi
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-2}"   # desired; will be clamped <= train_batch_size below
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent="${N_AGENT:-2}"

tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
val_before_train="${VAL_BEFORE_TRAIN:-false}"

search_url="${SEARCH_URL:-http://127.0.0.1:8003/search}"


max_turns="${MAX_TURNS:-2}"
timestamp=$(date +%Y%m%d_%H%M%S)
project_name="vrag"

# Resume training support
reward_mode="${REWARD_COMBINATION_MODE:-search_answer_citation_bbox_uniform}"

export REWARD_COMBINATION_MODE="$reward_mode"

engine="${ENGINE:-vllm}"

# -----------------------------
# Resume training controls
# -----------------------------
# Start fresh by default. Resume only when explicitly supplied via environment.
#
# You can still override via env:
#   RESUME_FROM_STEP_DIR / RESUME_CHECKPOINT_DIR / RESUME_EXPERIMENT_NAME
_DEFAULT_RUN_ROOT=""
_DEFAULT_STEP_DIR=""

# Optional resume training
# Or set via environment variables:
#   RESUME_FROM_STEP_DIR=/path/to/global_step_X
#   RESUME_CHECKPOINT_DIR=/path/to/run_root
#   RESUME_EXPERIMENT_NAME=original_experiment_name
resume_checkpoint_dir="${RESUME_CHECKPOINT_DIR:-${_DEFAULT_RUN_ROOT}}"
resume_from_step_dir="${RESUME_FROM_STEP_DIR:-${_DEFAULT_STEP_DIR}}"
resume_experiment_name="${RESUME_EXPERIMENT_NAME:-}"

# RM (reward model) config：脚本后面还会把这些传给 python，所以需要保留默认值
rm_workers_num="${RM_WORKERS_NUM:-1}"
rm_url="${RM_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions}"
# If DASHSCOPE_API_KEY is set, use it; otherwise allow RM_KEY override; else empty
rm_key="${DASHSCOPE_API_KEY:-${RM_KEY:-}}"
rm_model_name="${RM_MODEL_NAME:-qwen-max-latest}"



default_local_dir=""
save_dir_name=""
if [ -n "$resume_checkpoint_dir" ] || [ -n "$resume_from_step_dir" ]; then
    # Resume mode
    # Default: auto => load latest from default_local_dir/latest_checkpointed_iteration.txt
    resume_mode="auto"

    # Prefer explicit step dir if provided (forces resume from that snapshot)
    if [ -n "$resume_from_step_dir" ]; then
        # IMPORTANT:
        # - trainer.default_local_dir should be the RUN ROOT so new checkpoints are written alongside siblings
        # - trainer.resume_mode must contain 'global_step_' (can be absolute path) to force a specific restore point
        if [ ! -d "$resume_from_step_dir" ]; then
            echo "[ERROR] RESUME_FROM_STEP_DIR does not exist: $resume_from_step_dir" >&2
            exit 1
        fi
        default_local_dir="$(dirname "$resume_from_step_dir")"
        resume_mode="$resume_from_step_dir"
        save_dir_name="$(basename "$default_local_dir")"
        echo "[INFO] Resuming training FROM specific step dir: $resume_from_step_dir"
        echo "[INFO] Continuing to WRITE checkpoints into run root: $default_local_dir"
    else
        default_local_dir="$resume_checkpoint_dir"
        save_dir_name="$(basename "$default_local_dir")"
        echo "[INFO] Resuming training from run root: $default_local_dir"
        echo "[INFO] (Trainer will pick latest via latest_checkpointed_iteration.txt)"
    fi

    if [ -n "$resume_experiment_name" ]; then
        experiment_name="$resume_experiment_name"
        echo "[INFO] Using original experiment_name for wandb: $experiment_name"
    else
        experiment_name="VRAG_SFT_w_crop_${n_gpus}GPU_${max_turns}turns_${n_agent}agents_${reward_mode}_${timestamp}"
        echo "[WARN] No RESUME_EXPERIMENT_NAME provided. Wandb may create a new run name."
        echo "[WARN] To continue wandb plotting, set RESUME_EXPERIMENT_NAME to the original experiment name"
    fi
else
    # New training
    save_dir_name="${reward_mode}_${timestamp}"
    default_local_dir="checkpoints/${project_name}/${save_dir_name}"
    experiment_name="VRAG_SFT_w_crop_${n_gpus}GPU_${max_turns}turns_${n_agent}agents_${reward_mode}_${timestamp}"
    resume_mode="disable"
fi


export RAY_memory_usage_threshold=0.995

# -----------------------------
# Safety / sanity adjustments
# -----------------------------
# Ensure mini batch fits question batch times agent trajectories.
if [ "$((train_batch_size * n_agent))" -lt "$ppo_mini_batch_size" ]; then
  echo "[WARN] Reducing PPO mini batch to available trajectories."
  ppo_mini_batch_size=$((train_batch_size * n_agent))
fi

# Convert train_file(s) to Hydra list format if multiple files are provided
# Check if train_file contains spaces (multiple files)
if [[ "$train_file" == *" "* ]]; then
    # Multiple files: convert space-separated to Hydra list format [file1,file2,file3]
    train_files_list="[$(echo "$train_file" | sed 's/ /,/g')]"
else
    # Single file: keep as is (Hydra accepts single string or list)
    train_files_list="$train_file"
fi

# Print configuration
echo "=== Launch config ==="
echo "model_path: $model_path"
echo "train_file(s): $train_file"
echo "train_files_list: $train_files_list"
echo "val_file: $val_file"
echo "n_gpus: $n_gpus"
echo "train_batch_size: $train_batch_size"
echo "ppo_mini_batch_size: $ppo_mini_batch_size"
echo "ppo_micro_batch_size_per_gpu: $ppo_micro_batch_size_per_gpu"
echo "experiment_name: $experiment_name"
echo "save_dir_name: $save_dir_name"
echo "resume_mode: $resume_mode"
echo "default_local_dir: $default_local_dir"
echo "rm_key set: $( [ -z "$rm_key" ] && echo "NO" || echo "YES" )"
echo "reward_mode: $reward_mode"
echo "======================"

# -----------------------------
# Run the training module
# -----------------------------
export REWARD_COMBINATION_MODE="$reward_mode"
"$TRAIN_PY" -u scripts/run_grpo_smoke.py --config-name qwen3b_lowmem_smoke \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files_list" \
    data.val_files="$val_file" \
    data.train_batch_size="$train_batch_size" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-256}" \
    data.image_key=images \
    data.prompt_key=prompt \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ppo_micro_batch_size_per_gpu" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$log_prob_micro_batch_size_per_gpu" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$tensor_model_parallel_size" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.12}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent="$n_agent" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$log_prob_micro_batch_size_per_gpu" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager='rm' \
    reward_model.rm_workers_num="$rm_workers_num" \
    reward_model.rm_url="$rm_url" \
    reward_model.rm_key="$rm_key" \
    reward_model.rm_model_name="$rm_model_name" \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.default_local_dir="$default_local_dir" \
    trainer.n_gpus_per_node="$n_gpus" \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-1}" \
    trainer.resume_mode="$resume_mode" \
    trainer.val_before_train="$val_before_train" \
    retriever.url="$search_url" \
    max_turns="$max_turns" \
    "$@"
