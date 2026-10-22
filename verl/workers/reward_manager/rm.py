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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score, vrag
import verl.utils.relevance_score as rs
import torch
import json
import requests
import math
import numpy as np
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
import ast


DEFAULT_SYSTEM_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.

Please note that the generated answer may contain additional information beyond the reference answer.
"""

def parse_judge_response(response):
    pattern = r'<judge>(.*?)</judge>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            judge_str = match.group(0)
            if 'true' in judge_str or 'True' in judge_str:
                return True
            elif 'false' in judge_str or 'False' in judge_str:
                return False
            else:
                return None
        except Exception as e:
            return None
    else:
        return None

    
def dcg(relevance_scores):
    """
    计算折扣累积增益（DCG）
    :param relevance_scores: 一个列表，表示每个文档的相关性分数
    :return: DCG 值
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value

def ndcg(sorted_docs, golden_answer_list):
    """
    计算归一化折扣累积增益（NDCG）
    :param sorted_docs: 一个列表，表示已经排好序的文档
    :param golden_answer_list: 一个列表，表示所有相关文档（golden answers）
    :return: NDCG 值
    """
    # 将文档映射为相关性分数（在 golden_answer_list 中的文档为 1，否则为 0）
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    
    # 计算 DCG
    dcg_value = dcg(relevance_scores)
    
    # 计算 IDCG（理想情况下的 DCG，所有相关文档都排在前面）
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    
    # 防止分母为零
    if idcg_value == 0:
        return 0.0
    
    # 计算 NDCG
    ndcg_value = dcg_value / idcg_value
    return ndcg_value

def get_answer_from_predict_str(text):
    end_tag = '</answer>'
    start_tag = '<answer>'
    
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None  # 如果没有找到</answer>，返回None
    
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None  # 如果没有找到<answer>，返回None
    
    start_pos += len(start_tag)  # 跳过<answer>标签
    return text[start_pos:end_pos]


def parse_first_bbox_after_search(response_str: str) -> list | None:
    """Return the first <bbox> coordinates appearing after the <search> section."""
    search_end = re.search(r'</search>', response_str, re.IGNORECASE)
    rest = response_str[search_end.end():] if search_end else response_str
    bbox_match = re.search(r'<bbox>\s*\[([^\]]+)\]\s*</bbox>', rest, re.IGNORECASE)
    if not bbox_match:
        return None
    coords = re.findall(r'-?\d+', bbox_match.group(1))
    if len(coords) < 4:
        return None
    return [int(coords[i]) for i in range(4)]


def normalize_ground_truth_bbox(raw_bbox):
    """Normalize raw bbox into [x1,y1,x2,y2] ints or return None."""
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, str):
        try:
            raw_bbox = ast.literal_eval(raw_bbox)
        except Exception:
            return None
    if isinstance(raw_bbox, np.ndarray):
        raw_bbox = raw_bbox.tolist()
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        try:
            return [int(float(raw_bbox[i])) for i in range(4)]
        except Exception:
            return None
    return None


def compute_bbox_feedback_value(response_str: str,
                                extra_info: dict,
                                retrievaled_images: list,
                                crop_history: list) -> float | None:
    """Compute intersection / ground_truth bbox for the first bbox after <search>."""
    gt_bbox = normalize_ground_truth_bbox(
        extra_info.get('ground_truth_citation') or extra_info.get('bounding_box') if extra_info else None
    )
    if gt_bbox is None:
        return None
    predicted_bbox = parse_first_bbox_after_search(response_str)
    if predicted_bbox is None:
        return None
    try:
        crop_chain = vrag.build_crop_chain_from_extra_info(retrievaled_images, crop_history)
        converted_pred_bbox = vrag.convert_crop_coords_to_search_image(predicted_bbox, crop_chain)
        if converted_pred_bbox is None or len(converted_pred_bbox) != 4:
            return None
        return vrag.calculate_bbox_reward(converted_pred_bbox, gt_bbox)
    except Exception as e:
        print(f"[RM Debug] Failed to compute bbox feedback: {e}")
        return None


def is_refusal_answer(answer_text: str) -> bool:
    """
    Heuristic refusal detector for '<answer>...</answer>' content.
    We keep it simple and robust for training-time reward shaping.
    """
    if not answer_text:
        return False
    s = str(answer_text).strip().lower()
    # Common English refusals
    en_patterns = [
        r"don't have enough information",
        r"do not have enough information",
        r"not enough information",
        r"insufficient information",
        r"can't answer",
        r"cannot answer",
        r"unable to answer",
        r"i cannot determine",
        r"i can't determine",
        r"provided image",
        r"provided images",
    ]
   
    for pat in en_patterns:
        if re.search(pat, s):
            return True
    return False


def parse_bool_judge_response(response: str):
    """
    Parse '<judge>True</judge>' / '<judge>False</judge>' from LLM output.
    Returns: True / False / None
    """
    if response is None:
        return None
    m = re.search(r"<judge>\s*(true|false)\s*</judge>", str(response), re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip().lower() == "true"


class RMManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, eval_mode=False, rm_workers_num=10, rm_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", rm_key="EMPTY", rm_model_name="qwen-max-latest") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.eval_mode = eval_mode
        self.rm_workers_num = rm_workers_num
        self.rm_url = rm_url
        self.rm_key = rm_key
        self.rm_model_name = rm_model_name
        self.checkpoint_dir = None  # Will be set during validation
        # LLM-based refusal judge (optional; used in refusal-on-mismatch reward mode)
        self.refusal_judge_url = os.environ.get("REFUSAL_JUDGE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"))
        self.refusal_judge_key = os.environ.get("REFUSAL_JUDGE_KEY", os.environ.get("OPENAI_API_KEY", ""))
        self.refusal_judge_model = os.environ.get("REFUSAL_JUDGE_MODEL", "gpt-5")
        self.refusal_judge_enabled = os.environ.get("REFUSAL_JUDGE_ENABLED", "0").lower() in ("1", "true", "yes", "y")
        self.bbox_feedback_weight = float(os.environ.get("BBOX_FEEDBACK_WEIGHT", "0.2"))

    def _judge_refusal_with_llm(self, question: str, answer: str):
        """
        Ask an LLM (e.g., GPT-5) whether the answer is a refusal due to insufficient info from provided images.
        Return 1.0 for refusal, 0.0 otherwise. Falls back to heuristic if any error occurs.
        """
        try:
            if not self.refusal_judge_enabled or not self.refusal_judge_key:
                return 1.0 if is_refusal_answer(answer) else 0.0

            prompt = (
                "You are judging whether an assistant answer is a REFUSAL.\n"
                "A REFUSAL means the assistant explicitly says it cannot answer because there is not enough information "
                "from the provided image(s)/context.\n\n"
                f"Question:\n{question}\n\n"
                f"Answer:\n{answer}\n\n"
                "Reply with exactly one tag:\n"
                "<judge>True</judge> if it is a refusal, otherwise <judge>False</judge>."
            )

            # OpenAI Chat Completions-compatible payload
            payload = {
                "model": self.refusal_judge_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
            }
            # Timeout + retry (up to 3) to avoid transient network/endpoint slowness
            max_attempts = 3
            backoff_s = 0.5
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = requests.post(
                        self.refusal_judge_url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.refusal_judge_key}",
                        },
                        json=payload,
                        timeout=60,
                    )
                    # Retry on transient 5xx
                    if 500 <= getattr(resp, "status_code", 0) < 600:
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    resp.raise_for_status()
                    j = resp.json()
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts:
                        try:
                            time.sleep(backoff_s)
                        except Exception:
                            pass
                        backoff_s *= 2
                    else:
                        raise
            # OpenAI-style: choices[0].message.content
            content = None
            try:
                content = j["choices"][0]["message"]["content"]
            except Exception:
                # Some compatible providers may use different wrapper
                content = j.get("output", {}).get("content") or j.get("content")

            judge = parse_bool_judge_response(content)
            if judge is None:
                return 1.0 if is_refusal_answer(answer) else 0.0
            return 1.0 if judge else 0.0
        except Exception as e:
            print(f"[RM] Refusal LLM judge failed, fallback to heuristic. err={e}")
            return 1.0 if is_refusal_answer(answer) else 0.0
    
    def rm_score(self,data_eval_item):
        pay_load = {
            "model": self.rm_model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": DEFAULT_SYSTEM_TEMPLATE 
                        .replace("{query}", data_eval_item["query"]) 
                        .replace("{reference_answer}", data_eval_item["reference_answer"])  
                        .replace("{generated_answer}", data_eval_item["generated_answer"])
                }
            ]
        }
        max_retry = 20
        while True:
            if max_retry <= 0:
                return 0.0
            try:
                response = requests.post(
                    self.rm_url,
                    headers={
                        "Content-Type": "application/json",
                        "X-Auth-Key": self.rm_key,
                    },
                    json=pay_load
                )
                response.raise_for_status()
                result = response.json()
                judge_str = parse_judge_response(result['output']['content'])
                if judge_str is not None:
                    if judge_str:
                        return 1.0
                    else:
                        return 0.0
            except Exception as e:
                continue

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        
        # Debug: print which reward function is being called
        import inspect
        caller_frame = inspect.currentframe().f_back
        caller_info = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
        print(f'[RM Debug] RMManager.__call__ invoked from {caller_info}, eval_mode={self.eval_mode}, checkpoint_dir={self.checkpoint_dir}')

        # ------------------------------------------------------------
        # Refusal reward mode: only update the LAST turn/sample.
        #
        # In some multi-turn rollouts, a "dialog" is represented as a batch
        # of per-turn samples. When refusal shaping is enabled, we only want
        # to apply reward on the final turn, and keep previous turns unchanged.
        # ------------------------------------------------------------
        reward_mode_global = os.environ.get("REWARD_COMBINATION_MODE", "default")
        refusal_only_last_turn = reward_mode_global in {
            "search_answer_citation_refuse_on_mismatch",
            "refusal_only",
        }
        last_turn_index = len(data) - 1

        # ------------------------------------------------------------
        # Always attach refusal/search flags for logging (wandb/console),
        # even when model-based RM has already produced `rm_scores`.
        # This keeps refusal-related metrics visible in training logs.
        # ------------------------------------------------------------
        try:
            from verl.utils.reward_score import vrag
            metrics_enabled = os.environ.get("REFUSAL_METRICS_ENABLED", "1").lower() in ("1", "true", "yes", "y")
            if metrics_enabled:
                # Collect per-sample info first; judge refusals concurrently to speed up evaluation.
                items = []
                for i in range(len(data)):
                    data_item = data[i]
                    prompt_ids = data_item.batch['prompts']
                    prompt_length = prompt_ids.shape[-1]
                    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                    valid_response_ids = data_item.batch['responses'][:valid_response_length]
                    response_str = self.tokenizer.decode(valid_response_ids)

                    extra_info = data_item.non_tensor_batch.get('extra_info', None)
                    retrievaled_images = data_item.non_tensor_batch.get('retrievaled_images', [])
                    # Normalize retrievaled_images possibly stored as numpy object
                    try:
                        if hasattr(retrievaled_images, 'tolist'):
                            retrievaled_images = retrievaled_images.tolist()
                    except Exception:
                        pass

                    search_correct = False
                    try:
                        search_correct = vrag.evaluate_search_correctness(extra_info or {}, retrievaled_images)
                    except Exception:
                        search_correct = False

                    ans_for_refusal = get_answer_from_predict_str(response_str)
                    if ans_for_refusal is None:
                        ans_for_refusal = response_str
                    q_for_refusal = (extra_info.get('question') if isinstance(extra_info, dict) else '') or ''

                    items.append((i, search_correct, q_for_refusal, ans_for_refusal))

                # Threaded refusal judging (optional but enabled by default when workers > 1).
                try:
                    workers_env = os.environ.get("REFUSAL_JUDGE_WORKERS", "").strip()
                    refusal_workers = int(workers_env) if workers_env else int(getattr(self, "rm_workers_num", 1))
                except Exception:
                    refusal_workers = int(getattr(self, "rm_workers_num", 1))

                def _judge_one(tup):
                    _i, _search_correct, _q, _a = tup
                    return _i, float(self._judge_refusal_with_llm(_q, _a or ""))

                refusal_flags = {}
                if refusal_workers > 1 and len(items) > 0:
                    with ThreadPoolExecutor(max_workers=refusal_workers) as pool:
                        for _i, _rf in pool.map(_judge_one, items):
                            refusal_flags[_i] = _rf
                else:
                    for tup in items:
                        _i, _rf = _judge_one(tup)
                        refusal_flags[_i] = _rf

                # Write back metrics
                for i, search_correct, _, _ in items:
                    data_item = data[i]
                    if 'reward_model' not in data_item.non_tensor_batch or not isinstance(data_item.non_tensor_batch.get('reward_model'), dict):
                        data_item.non_tensor_batch['reward_model'] = {}
                    data_item.non_tensor_batch['reward_model']['search_correct'] = bool(search_correct)
                    data_item.non_tensor_batch['reward_model']['refusal_flag'] = float(refusal_flags.get(i, 0.0))
        except Exception as e:
            print(f"[RM] Warning: failed to attach refusal metrics: {e}")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if refusal_only_last_turn and last_turn_index >= 0:
                rm_scores = data.batch['rm_scores'].clone()
                # Expect shape [bs, seq], but handle other shapes defensively.
                try:
                    if rm_scores.dim() >= 2:
                        rm_scores[:last_turn_index].zero_()
                    else:
                        rm_scores[:last_turn_index] = 0
                except Exception:
                    # If any shape mismatch occurs, fall back to returning original rm_scores.
                    return data.batch['rm_scores']
                return rm_scores
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        data_eval = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            generated_answer = get_answer_from_predict_str(self.tokenizer.decode(valid_response_ids))
            if generated_answer is None:
                generated_answer = 'Please Judge False'
            data_eval.append(dict(
                query = extra_info['question'],
                generated_answer = generated_answer,
                reference_answer = data_item.non_tensor_batch['reward_model']['ground_truth']
            ))

        data_to_be_eval = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Refusal reward mode: only apply reward on the final turn/sample.
            if refusal_only_last_turn and i != last_turn_index:
                continue

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            from project_paths import project_path
            output_file = project_path("outputs/debug/check_output.txt")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write("Prompt:\n")
                f.write(prompt_str + "\n\n")
                f.write("Response:\n")
                f.write(response_str + "\n\n")
                f.write("Ground Truth:\n")
                f.write(str(ground_truth) + "\n")
                f.write("="*50 + "\n")  


            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # For wiki-visa dataset, skip reward model evaluation
            if score >0.0 and data_source != 'wiki-visa':
                data_to_be_eval.append(data_eval[i])
        
        if len(data_to_be_eval) > 0:
            
            if self.rm_workers_num > 1:
                with ThreadPoolExecutor(max_workers=self.rm_workers_num) as pool:
                    eval_results = list(pool.map(self.rm_score, data_to_be_eval))
            else:
                eval_results = [self.rm_score(data_to_be_eval[i]) for i in range(len(data_to_be_eval))]

            # request_data_to_be_eval = dict(
            #     bs=300,
            #     prompts=data_to_be_eval
            # )
            # prompts_json = json.dumps(request_data_to_be_eval)
            # print("=====================eval model start=====================")
            # response = requests.post(self.rm_url, json=prompts_json)
            # eval_results = response.json()
            # print("=====================eval model end=====================")
            
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # For wiki-visa dataset, use comprehensive reward calculation
            # treat all data_source as wiki-visa for reward calculation
            if True:
                # ========== 格式检查 ==========
                # 检查必须包含的标签：search, bbox, answer, citation
                answer_pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)
                search_pattern = re.compile(r'<search>.*?</search>', re.DOTALL)
                bbox_pattern = re.compile(r'<bbox>.*?</bbox>', re.DOTALL)
                citation_pattern = re.compile(r'<citation>.*?</citation>', re.DOTALL)
                
                has_answer = bool(answer_pattern.search(response_str))
                has_search = bool(search_pattern.search(response_str))
                has_bbox = bool(bbox_pattern.search(response_str))
                has_citation = bool(citation_pattern.search(response_str))

                # ========== 统一计算：search_correct / ndcg / citation_iou / bbox_reward ==========
                from verl.utils.reward_score import vrag
                retrievaled_images = data_item.non_tensor_batch.get('retrievaled_images', [])
                crop_history = data_item.non_tensor_batch.get('crop_history', [])

                # Normalize retrievaled_images possibly stored as numpy object
                try:
                    if hasattr(retrievaled_images, 'tolist'):
                        retrievaled_images = retrievaled_images.tolist()
                except Exception:
                    pass

                # ========== Refusal flag for metrics & reward penalty ==========
                # Use the same refusal judge as training-time shaping:
                # - If REFUSAL_JUDGE_ENABLED=1 and key is provided -> LLM judge
                # - Otherwise -> heuristic fallback inside _judge_refusal_with_llm
                ans_for_refusal = get_answer_from_predict_str(response_str)
                if ans_for_refusal is None:
                    ans_for_refusal = response_str
                q_for_refusal = (extra_info.get('question') if extra_info else '') or ''
                refusal_flag = float(self._judge_refusal_with_llm(q_for_refusal, ans_for_refusal or ""))

                # Reward mode selection (early).
                reward_mode = os.environ.get("REWARD_COMBINATION_MODE", "default")

                # Search correctness: required for conditional refusal mode.
                search_correct = False
                try:
                    search_correct = vrag.evaluate_search_correctness(extra_info or {}, retrievaled_images)
                except Exception as e:
                    print(f"[RM] Warning: Failed to evaluate search correctness: {e}")
                    search_correct = False

                # Conditional refusal mode:
                # - If search is correct -> reward answer correctness (answer_score from compute_score)
                # - Else -> reward refusal correctness (refusal_flag)
                # We still require at least an <answer> tag to be considered well-formed.
                if reward_mode == "refusal_only":
                    answer_score = score  # compute_score output
                    if not has_answer:
                        score = 0.0
                    else:
                        if search_correct:
                            # If retrieval is correct, we want the model to answer (not refuse).
                            # Penalize refusal under successful retrieval.
                            try:
                                penalty = float(os.environ.get("REFUSAL_PENALTY_WHEN_SEARCH_CORRECT", "1.0"))
                            except Exception:
                                penalty = 1.0
                            score = float(answer_score) - penalty * float(refusal_flag)
                        else:
                            # If retrieval is incorrect, encourage refusal.
                            score = float(refusal_flag)

                # Store per-sample stats for later aggregation/logging (trainer will read these)
                if 'reward_model' not in data_item.non_tensor_batch or not isinstance(data_item.non_tensor_batch.get('reward_model'), dict):
                    data_item.non_tensor_batch['reward_model'] = {}
                # Populate metrics
                data_item.non_tensor_batch['reward_model']['search_correct'] = bool(search_correct)
                data_item.non_tensor_batch['reward_model']['refusal_flag'] = float(refusal_flag)
                data_item.non_tensor_batch['reward_model']['has_answer_tag'] = bool(has_answer)
                data_item.non_tensor_batch['reward_model']['has_search_tag'] = bool(has_search)

                # 1) answer_score (default correctness score from compute_score)
                answer_score = score

                # 2) search_ndcg_value
                search_ndcg_value = 0.0
                reference_page = extra_info.get('reference_page', None) if extra_info else None
                file_name = extra_info.get('file_name', None) if extra_info else None
                if retrievaled_images and reference_page:
                    try:
                        retrievaled_images_basename_list = [
                            os.path.basename(str(item).rstrip('/')).split(".jpg")[0]
                            for item in retrievaled_images
                        ]
                        if hasattr(reference_page, 'tolist'):
                            reference_page_list = reference_page.tolist()
                        elif isinstance(reference_page, list):
                            reference_page_list = reference_page
                        else:
                            reference_page_list = [reference_page]

                        # Legacy PDF-page IDs and current Wiki-VISA raw page IDs.
                        reference_images_basename_list = [
                            f'{file_name.split(".pdf")[0]}_{page}' if file_name else str(page)
                            for page in reference_page_list
                        ]
                        search_ndcg_value = ndcg(retrievaled_images_basename_list, reference_images_basename_list)
                    except Exception as e:
                        print(f"[RM] Warning: Failed to calculate NDCG for wiki-visa: {e}")
                        search_ndcg_value = 0.0

                # 3) citation_iou_value
                citation_iou_value = 0.0
                if extra_info:
                    try:
                        citation_iou_value = vrag.evaluate_citation_iou(
                            response_str, extra_info, retrievaled_images, crop_history
                        )
                    except Exception as e:
                        print(f"[RM] Warning: Failed to calculate citation IoU for wiki-visa: {e}")
                        citation_iou_value = 0.0

                # 4) bbox_reward_value
                bbox_reward_value = 0.0
                if extra_info:
                    try:
                        bbox_reward_value = vrag.evaluate_bbox_reward(
                            response_str, extra_info, retrievaled_images, crop_history
                        )
                    except Exception as e:
                        print(f"[RM] Warning: Failed to calculate bbox_reward for wiki-visa: {e}")
                        bbox_reward_value = 0.0

                # ========== 加权组合 ==========
                # reward_mode is already resolved above

                # ------------------------------------------------------------
                # New mode: search-r1
                # - RM reward is EXACTLY answer accuracy (EM on <answer> content)
                # - No search/citation/bbox/relevance/refusal shaping is mixed in
                # ------------------------------------------------------------
                if reward_mode == "search-r1":
                    try:
                        score = float(vrag.compute_answer_acc(response_str, ground_truth, extra_info))
                    except Exception as e:
                        print(f"[RM] Warning: search-r1 compute_answer_acc failed: {e}")
                        score = 0.0

                # 新模式：当“所有检索图片都不匹配”（search_correct=False）时，启用拒答 reward
                # - 放宽格式检查：不要求 citation（也默认不再强制 bbox）
                # - reward = 0.2 * search + 0.8 * refusal
                elif reward_mode == "search_answer_citation_refuse_on_mismatch":
                    if not (has_answer and has_search):
                        print(f"[RM Debug] Format check failed (refuse_on_mismatch): answer={has_answer}, search={has_search}")
                        score = 0.0
                    else:
                        if not search_correct:
                            # retrieval fail: encourage refusal
                            score = 0.6 * search_ndcg_value + 0.4 * refusal_flag
                        else:
                            # search 匹配时：沿用原有严格格式与 reward（search + answer + citation）
                            # plus: add refusal penalty if model still refuses under successful retrieval
                            if not (has_answer and has_search and has_bbox and has_citation):
                                print(f"[RM Debug] Format check failed: answer={has_answer}, search={has_search}, bbox={has_bbox}, citation={has_citation}")
                                score = 0.0
                            else:
                                score = 0.2 * search_ndcg_value + 0.4 * answer_score + 0.4 * citation_iou_value
                elif reward_mode == "with_relevance":
                    print(f"[RM Debug] Calculating relevance score... (reward_mode={reward_mode})")
                    # Debug extra_info keys
                    if extra_info:
                        print(f"[RM Debug] extra_info keys: {list(extra_info.keys())}")
                        # 如果retrievaled_images丢失，尝试从data_item.non_tensor_batch恢复
                        if 'retrievaled_images' not in extra_info and 'retrievaled_images' in data_item.non_tensor_batch:
                            extra_info['retrievaled_images'] = data_item.non_tensor_batch['retrievaled_images']
                            print(f"[RM Debug] Recovered retrievaled_images from data_item: {len(extra_info['retrievaled_images'])} items")
                        
                        if 'retrievaled_images' in extra_info:
                            print(f"[RM Debug] retrievaled_images in extra_info: {len(extra_info['retrievaled_images'])} items")
                        else:
                            print("[RM Debug] 'retrievaled_images' NOT found in extra_info")
                    
                    relevance_val = 0.0
                    try:
                        rm_config = {
                            'rm_url': self.rm_url,
                            'rm_key': self.rm_key,
                            'rm_model_name': self.rm_model_name
                        }
                        relevance_val = rs.evaluate_relevance(response_str, extra_info, rm_config)
                        print(f"[RM Debug] Relevance score: {relevance_val}")
                        
                        # 将relevance score存储到non_tensor_batch中以便后续统计
                        if 'reward_model' not in data_item.non_tensor_batch:
                            data_item.non_tensor_batch['reward_model'] = {}
                        data_item.non_tensor_batch['reward_model']['relevance_score'] = relevance_val
                        
                    except Exception as e:
                        print(f"[RM] Relevance eval failed: {e}")
                        relevance_val = 0.0
                        if 'reward_model' not in data_item.non_tensor_batch:
                            data_item.non_tensor_batch['reward_model'] = {}
                        data_item.non_tensor_batch['reward_model']['relevance_score'] = 0.0
                    # 权重分配：search 0.1, answer 0.4, citation 0.3, relevance 0.2
                    score = 0.1 * search_ndcg_value + 0.4 * answer_score + 0.3 * citation_iou_value + 0.2 * relevance_val
                elif reward_mode == "search_answer":
                    score = 0.2 * search_ndcg_value + 0.8 * answer_score
                elif reward_mode == "search_answer_citation":
                    if not (has_answer and has_search and has_bbox and has_citation):
                        print(f"[RM Debug] Format check failed: answer={has_answer}, search={has_search}, bbox={has_bbox}, citation={has_citation}")
                        score = 0.0
                    else:
                        score = 0.2 * search_ndcg_value + 0.4 * answer_score + 0.4 * citation_iou_value
                elif reward_mode == "search_answer_citation_with_bbox_feedback":
                    if not (has_answer and has_search and has_bbox and has_citation):
                        print(f"[RM Debug] Format check failed: answer={has_answer}, search={has_search}, bbox={has_bbox}, citation={has_citation}")
                        score = 0.0
                    else:
                        # 权重策略：
                        # 1. 基础分：0.2*search + 0.4*answer
                        # 2. 如果检索正确：+0.3*citation + 0.1*bbox
                        # 3. 如果检索错误：citation和bbox部分均不得分
                        w_search = 0.2
                        w_answer = 0.4
                        w_citation = 0.3
                        w_bbox = 0.1
                        
                        score = w_search * search_ndcg_value + w_answer * answer_score
                        
                        if search_correct:
                            score += w_citation * citation_iou_value
                            
                            bbox_val = compute_bbox_feedback_value(response_str, extra_info or {}, retrievaled_images, crop_history)
                            if bbox_val is not None:
                                bbox_reward = float(bbox_val)
                                data_item.non_tensor_batch.setdefault('reward_model', {})['bbox_feedback'] = bbox_reward
                                score += w_bbox * bbox_reward
                elif reward_mode == "search_answer_citation_bbox_uniform":
                    if not (has_answer and has_search and has_bbox and has_citation):
                        print(f"[RM Debug] Format check failed: answer={has_answer}, search={has_search}, bbox={has_bbox}, citation={has_citation}")
                        score = 0.0
                    else:
                        # 统一权重策略：不区分检索是否成功
                        # score = 0.1 * search + 0.4 * answer + 0.4 * citation + 0.1 * bbox
                        score = 0.15 * search_ndcg_value + 0.4 * answer_score + 0.35 * citation_iou_value + 0.1 * bbox_reward_value
                elif reward_mode == "bbox_reward":
                    if not (has_answer and has_search and has_bbox and has_citation):
                        print(f"[RM Debug] Format check failed: answer={has_answer}, search={has_search}, bbox={has_bbox}, citation={has_citation}")
                        score = 0.0
                    else:
                        # 权重分配：search 0.2, answer 0.4, citation 0.2, bbox_reward 0.2
                        score = 0.2 * search_ndcg_value + 0.4 * answer_score + 0.2 * citation_iou_value + 0.2 * bbox_reward_value
                else:
                    # fallback 和 default 保持原来 mix
                    if not (has_answer and has_search and has_bbox and has_citation):
                        print(f"[RM Debug] Format check failed: answer={has_answer}, search={has_search}, bbox={has_bbox}, citation={has_citation}")
                        score = 0.0
                    else:
                        score = 0.2 * search_ndcg_value + 0.4 * answer_score + 0.4 * citation_iou_value

            # Write answer to checkpoint directory in eval mode (for all datasets including wiki-visa)
            if self.eval_mode and self.checkpoint_dir is not None:
                try:
                    # Extract question and predicted answer
                    question = extra_info.get('question', '') if extra_info else ''
                    predicted_answer = get_answer_from_predict_str(response_str)
                    if predicted_answer is None:
                        predicted_answer = response_str  # Fallback to full response if no answer tag
                    
                    log_entry = {
                        'question': question,
                        'predicted_answer': predicted_answer,
                        'ground_truth': ground_truth,
                        'score': float(score),
                        'full_response': response_str,
                        'data_source': data_source  # 添加数据源信息
                    }
                    
                    # Ensure directory exists
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    
                    # Write to checkpoint directory
                    rm_eval_log_path = os.path.join(self.checkpoint_dir, 'rm_eval_log.jsonl')
                    with open(rm_eval_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                        f.flush()  # 立即刷新到磁盘
                    if i == 0:  # 只在第一个样本时打印一次
                        print(f'[RM Eval] Writing to checkpoint log: {rm_eval_log_path} (data_source={data_source}, eval_mode={self.eval_mode})')
                except Exception as e:
                    print(f"Error writing rm_eval log for sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if i == 0:  # 只在第一个样本时打印一次
                    print(f'[RM Eval] Skipping log write: eval_mode={self.eval_mode}, checkpoint_dir={self.checkpoint_dir}, data_source={data_source}')

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
