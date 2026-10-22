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

import re
import ast
import math
import os
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# 缩放参数（与训练时generation.py和推理时inference_vrag_20251129.py一致）
MAX_PIXELS = 1536 * 28 * 28  # 1,204,416
MIN_PIXELS = 768 * 28 * 28   # 602,208


def process_image_size(original_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    计算图像经过process_image处理后的尺寸（与evaluate_inference.py一致）
    
    Args:
        original_size: (width, height) 原始图像尺寸
    
    Returns:
        (width, height) 处理后的图像尺寸
    """
    original_w, original_h = original_size
    original_pixels = original_w * original_h
    
    # 如果图像太大：缩小
    if original_pixels > MAX_PIXELS:
        resize_factor = math.sqrt(MAX_PIXELS / original_pixels)
        new_width = int(original_w * resize_factor)
        new_height = int(original_h * resize_factor)
        return (new_width, new_height)
    
    # 如果图像太小：放大
    elif original_pixels < MIN_PIXELS:
        resize_factor = math.sqrt(MIN_PIXELS / original_pixels)
        new_width = int(math.ceil(original_w * resize_factor))
        new_height = int(math.ceil(original_h * resize_factor))
        # 如果取整后仍然小于MIN_PIXELS，再次调整
        new_pixels = new_width * new_height
        if new_pixels < MIN_PIXELS:
            scale_adjust = math.sqrt(MIN_PIXELS / new_pixels)
            new_width = int(math.ceil(new_width * scale_adjust))
            new_height = int(math.ceil(new_height * scale_adjust))
        return (new_width, new_height)
    
    # 不需要缩放
    return original_size


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """
    获取图像尺寸
    
    Args:
        image_path: 图像路径
    
    Returns:
        (width, height) 或 None
    """
    try:
        if image_path and os.path.exists(image_path):
            with Image.open(image_path) as img:
                return img.size
    except Exception:
        pass
    return None


def build_crop_chain_from_extra_info(retrievaled_images: List[str], crop_history: List[Dict]) -> List[Dict]:
    """
    从retrievaled_images和crop_history构建裁剪链（适配evaluate_inference.py的格式）
    
    Args:
        retrievaled_images: 检索到的图像路径列表
        crop_history: 裁剪历史，每个元素包含 {'bbox': [x1,y1,x2,y2], 'image_path': str, 'step': int}
    
    Returns:
        crop_chain: 裁剪链，包含每一步的源图像、bbox和结果图像
    """
    crop_chain = []
    last_raw_size = None
    last_processed_size = None
    
    # 添加检索步骤
    for img_path in retrievaled_images:
        raw_size = get_image_size(img_path)
        processed_size = process_image_size(raw_size) if raw_size else None
        crop_chain.append({
            'step': len(crop_chain),
            'type': 'search',
            'source_image': img_path,
            'bbox': None,
            'result_image': img_path,
            'raw_size': raw_size,
            'processed_size': processed_size
        })
        last_raw_size = raw_size
        last_processed_size = processed_size
    
    # 添加裁剪步骤
    for crop_info in crop_history:
        bbox = crop_info.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        source_image = crop_chain[-1]['result_image'] if crop_chain else None
        image_path = crop_info.get('image_path')
        raw_size = None
        if image_path:
            raw_size = get_image_size(image_path)
        if raw_size is None and last_raw_size and last_processed_size:
            parent_proc_size = last_processed_size or process_image_size(last_raw_size)
            if parent_proc_size and parent_proc_size[0] > 0 and parent_proc_size[1] > 0:
                parent_raw_w, parent_raw_h = last_raw_size
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                scale_x = parent_raw_w / parent_proc_size[0]
                scale_y = parent_raw_h / parent_proc_size[1]
                raw_width = max(1, int(bbox_width * scale_x))
                raw_height = max(1, int(bbox_height * scale_y))
                raw_size = (raw_width, raw_height)
        processed_size = process_image_size(raw_size) if raw_size else None
        crop_chain.append({
            'step': len(crop_chain),
            'type': 'crop',
            'source_image': source_image,
            'bbox': bbox,
            'result_image': image_path,
            'raw_size': raw_size,
            'processed_size': processed_size
        })
        if raw_size:
            last_raw_size = raw_size
            last_processed_size = processed_size
    
    return crop_chain


def convert_crop_coords_to_search_image(citation_coords: List[int], crop_chain: List[Dict]) -> Optional[List[int]]:
    """
    将citation坐标从裁剪坐标系转换到最后一次检索图像坐标系（原始尺寸）
    与evaluate_inference.py的逻辑一致
    
    Args:
        citation_coords: [x1,y1,x2,y2] 在最后一次裁剪并缩放后的图像上的坐标
        crop_chain: 裁剪链信息
    
    Returns:
        [x1,y1,x2,y2] 在最后一次检索图像原始尺寸上的坐标，如果转换失败返回None
    """
    if not citation_coords or not crop_chain:
        return citation_coords

    # 找到最后一次检索步骤
    search_steps = [step for step in crop_chain if step['type'] == 'search']
    if not search_steps:
        return citation_coords

    last_search_step = search_steps[-1]
    last_search_step_idx = last_search_step['step']

    # 获取最后一次检索之后的所有裁剪步骤（从后往前）
    crop_steps_after_search = [step for step in crop_chain 
                                if step['type'] == 'crop' and step['step'] > last_search_step_idx]
    crop_steps_after_search.reverse()  # 从最后一次裁剪开始往前

    def _resolve_sizes(step: Dict) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        raw_size = step.get('raw_size')
        processed_size = step.get('processed_size')
        if raw_size is None:
            result_path = step.get('result_image')
            if result_path:
                raw_size = get_image_size(result_path)
        if processed_size is None and raw_size:
            processed_size = process_image_size(raw_size)
        return raw_size, processed_size

    # 当前坐标（相对于最后一次裁剪并缩放后的图像）
    current_coords = citation_coords.copy()

    # 从最后一次裁剪开始，逐步反向转换到检索图像
    for step_idx, step in enumerate(crop_steps_after_search):
        if not step.get('bbox'):
            continue
        
        cropped_saved_size, cropped_processed_size = _resolve_sizes(step)
        if not cropped_saved_size or not cropped_processed_size:
            return None
        
        cropped_w_saved, cropped_h_saved = cropped_saved_size
        cropped_w_proc, cropped_h_proc = cropped_processed_size
        
        # 计算缩放比例（从模型看到的缩放后图像到保存的裁剪图像）
        scale_x = cropped_w_saved / cropped_w_proc if cropped_w_proc > 0 else 1.0
        scale_y = cropped_h_saved / cropped_h_proc if cropped_h_proc > 0 else 1.0
        
        # 将坐标从模型看到的缩放后图像转换到保存的裁剪图像（原始尺寸）
        current_coords = [
            int(current_coords[0] * scale_x),
            int(current_coords[1] * scale_y),
            int(current_coords[2] * scale_x),
            int(current_coords[3] * scale_y)
        ]
        
        # 步骤2: 将坐标从裁剪区域转换到上一级图像
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = step['bbox']
        
        if step_idx < len(crop_steps_after_search) - 1:
            prev_step = crop_steps_after_search[step_idx + 1]
        else:
            prev_step = last_search_step
        prev_raw_size, prev_processed_size = _resolve_sizes(prev_step)
        if not prev_raw_size or not prev_processed_size:
            return None
        prev_w_saved, prev_h_saved = prev_raw_size
        prev_w_proc, prev_h_proc = prev_processed_size
        
        prev_scale_x = prev_w_saved / prev_w_proc if prev_w_proc > 0 else 1.0
        prev_scale_y = prev_h_saved / prev_h_proc if prev_h_proc > 0 else 1.0
        
        bbox_orig_x1 = int(bbox_x1 * prev_scale_x)
        bbox_orig_y1 = int(bbox_y1 * prev_scale_y)
        
        # 当前坐标是相对于裁剪区域左上角的，需要加上bbox在原始图像上的偏移
        current_coords = [
            bbox_orig_x1 + current_coords[0],
            bbox_orig_y1 + current_coords[1],
            bbox_orig_x1 + current_coords[2],
            bbox_orig_y1 + current_coords[3]
        ]
    
    # 如果没有裁剪步骤，citation坐标直接相对于检索图像的缩放后图像
    if not crop_steps_after_search:
        search_raw_size, search_processed_size = _resolve_sizes(last_search_step)
        if search_raw_size and search_processed_size:
            search_w_orig, search_h_orig = search_raw_size
            search_w_proc, search_h_proc = search_processed_size
            search_scale_x = search_w_orig / search_w_proc if search_w_proc > 0 else 1.0
            search_scale_y = search_h_orig / search_h_proc if search_h_proc > 0 else 1.0
            current_coords = [
                int(current_coords[0] * search_scale_x),
                int(current_coords[1] * search_scale_y),
                int(current_coords[2] * search_scale_x),
                int(current_coords[3] * search_scale_y)
            ]
    
    return current_coords


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

def calculate_anls(gold_labels, prediction, threshold=0.7):
    import numpy as np
    from Levenshtein import distance as levenshtein_distance
    max_scores = []
    for gold_label in gold_labels:
        ld = levenshtein_distance(prediction, gold_label)
        max_len = max(len(prediction), len(gold_label))
        if max_len == 0:
            nld = 0.0
        else:
            nld = ld / max_len
        if nld < threshold:
            score = 1 - nld
        else:
            score = 0.0
        max_scores.append(score)
    return max(max_scores)
def compute_score(predict_str: str, ground_truth: str, extra_info) -> float:
    predict_str = remove_text_between_tags(predict_str)
    format_reward_value=compute_format_reward_only(predict_str)

    if format_reward_value==1.0:
        answer = get_answer_from_predict_str(predict_str)
        if answer is None:
            return 0.0
        anls_score = calculate_anls([ground_truth], answer, 0.5)
        return anls_score
    else:
        return 0.0

def remove_text_between_tags(text):
    # 使用正则表达式匹配<|im_start|>和<|im_end|>之间的内容以及这两个标签
    pattern = r'<\|im_start\|>user.*?<\|im_end\|>'
    # 替换匹配到的内容为空字符串
    result = re.sub(pattern, '', text)
    return result
def compute_format_reward_only(predict_str: str, ground_truth: str, extra_info) -> float:
    predict_str = remove_text_between_tags(predict_str)
    answer_pattern = re.compile(r'<answer>.*</answer>', re.DOTALL)
    search_pattern = re.compile(r'<search>.*</search>', re.DOTALL)
    answer_match = re.search(answer_pattern, predict_str)
    search_match = re.search(search_pattern, predict_str)
    return 1.0 if answer_match and search_match else 0.0

def normalize_answer(text):
    """
    Normalize answer for exact match comparison.
    - Convert to lowercase
    - Remove leading/trailing whitespace
    - Remove extra whitespace
    - Remove common punctuation
    """
    if text is None:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common punctuation (optional, can be adjusted)
    text = re.sub(r'[.,;:!?\'"]', '', text)
    return text

def compute_answer_acc(predict_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute exact match (EM) accuracy for wiki-visa dataset.
    
    Args:
        predict_str: The model's response string
        ground_truth: The ground truth answer
        extra_info: Optional extra information (not used in EM)
    
    Returns:
        float: 1.0 if exact match, 0.0 otherwise
    """
    # Remove text between tags (like <|im_start|>user...<|im_end|>)
    predict_str = remove_text_between_tags(predict_str)
    
    # Extract answer from <answer> tags
    answer = get_answer_from_predict_str(predict_str)
    
    if answer is None:
        return 0.0
    
    # Normalize both answers for comparison
    normalized_pred = normalize_answer(answer)
    normalized_gt = normalize_answer(ground_truth)
    
    # Exact match comparison
    if normalized_pred == normalized_gt:
        return 1.0
    else:
        return 0.0


def get_citation_from_predict_str(text: str) -> Optional[List[int]]:
    """
    Extract citation coordinates from text.
    
    Args:
        text: Text containing citation tag, e.g., "<citation>[100, 100, 200, 200]</citation>"
        
    Returns:
        List of 4 integers [x1, y1, x2, y2] if found, None otherwise
    """
    citation_match = re.search(r'<citation>\s*\[([^\]]+)\]\s*</citation>', text)
    if citation_match:
        try:
            citation = ast.literal_eval(f"[{citation_match.group(1)}]")
            if isinstance(citation, list) and len(citation) == 4:
                return citation
        except (ValueError, SyntaxError):
            pass
    return None


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate IoU between two bounding boxes"""
    if not bbox1 or not bbox2:
        return 0.0

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_bbox_reward(pred_bbox: List[int], gt_bbox: List[int]) -> float:
    """
    Calculate bbox reward as intersection / ground_truth_area.
    
    Args:
        pred_bbox: Predicted bounding box [x1, y1, x2, y2]
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2]
    
    Returns:
        float: intersection / ground_truth_area, range [0.0, 1.0+]
        (can exceed 1.0 if prediction is larger than ground truth)
    """
    if not pred_bbox or not gt_bbox:
        return 0.0
    
    if len(pred_bbox) != 4 or len(gt_bbox) != 4:
        return 0.0
    
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    
    # Calculate intersection
    x_left = max(x1_pred, x1_gt)
    y_top = max(y1_pred, y1_gt)
    x_right = min(x2_pred, x2_gt)
    y_bottom = min(y2_pred, y2_gt)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate ground truth area
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    if gt_area == 0:
        return 0.0
    
    return intersection / gt_area


def evaluate_search_correctness(extra_info: Dict, retrievaled_images: List[str] = None) -> bool:
    """
    Evaluate if the search retrieved the correct images.
    
    Args:
        extra_info: Dictionary containing reference_page information
        retrievaled_images: List of retrieved image paths (optional, if not provided, will try to extract from extra_info)
    
    Returns:
        bool: True if search is correct, False otherwise
    """
    if not extra_info:
        return False
    
    reference_page = extra_info.get('reference_page', None)
    if reference_page is None:
        return False
    
    # Convert reference_page to list of strings (handle ndarray/list/str)
    if hasattr(reference_page, 'tolist'):
        try:
            reference_page = reference_page.tolist()
        except Exception:
            reference_page = [reference_page]
    if isinstance(reference_page, str):
        reference_page = [reference_page]
    elif not isinstance(reference_page, list):
        reference_page = [str(reference_page)]
    reference_page_ids = set(str(page_id) for page_id in reference_page if page_id is not None)
    # Also include file_name as a reference id candidate if present
    file_name = extra_info.get('file_name')
    if file_name is not None:
        reference_page_ids.add(str(file_name))
    
    # Get retrieved image IDs
    if retrievaled_images is None:
        retrievaled_images = extra_info.get('retrievaled_images', [])
    
    if not retrievaled_images:
        return False
    
    # Extract image IDs from paths (robust parsing)
    search_image_ids = []
    for img_path in retrievaled_images:
        candidates = set()
        path_val = None
        if isinstance(img_path, dict):
            # direct image_id
            image_id = img_path.get('image_id')
            if image_id is not None:
                candidates.add(str(image_id))
            path_val = img_path.get('image_file') or img_path.get('path') or ''
        else:
            path_val = img_path
        # From path: basename stem
        try:
            import os
            base = os.path.basename(str(path_val))
            stem = base.rsplit('.', 1)[0] if '.' in base else base
            if stem:
                candidates.add(stem)
            # numeric substrings inside stem
            for n in re.findall(r'\\d+', stem):
                candidates.add(n)
        except Exception:
            pass
        search_image_ids.extend(candidates)
    
    # Check if any retrieved image matches reference_page (equality or containment)
    if search_image_ids and reference_page_ids:
        if any(sid in reference_page_ids for sid in search_image_ids):
            return True
        for sid in search_image_ids:
            for ref in reference_page_ids:
                if ref in sid:
                    return True
    
    return False


def evaluate_answer_correctness(predict_str: str, ground_truth: str) -> Dict[str, bool]:
    """
    Evaluate answer correctness with both exact match and containment match.
    
    Args:
        predict_str: The model's response string
        ground_truth: The ground truth answer
    
    Returns:
        Dict with 'exact_correct' and 'containment_correct' keys
    """
    predict_str = remove_text_between_tags(predict_str)
    answer = get_answer_from_predict_str(predict_str)
    
    result = {
        'exact_correct': False,
        'containment_correct': False,
        'has_answer': answer is not None
    }
    
    if answer is None:
        return result
    
    # Normalize answers
    ref_clean = re.sub(r'\s+', ' ', ground_truth.lower().strip())
    pred_clean = re.sub(r'\s+', ' ', answer.lower().strip())
    
    # Exact match
    result['exact_correct'] = ref_clean == pred_clean
    
    # Containment match (reference answer contained in predicted answer)
    result['containment_correct'] = ref_clean in pred_clean
    
    return result


def evaluate_bbox_reward(predict_str: str, extra_info: Dict,
                         retrievaled_images: List[str] = None,
                         crop_history: List[Dict] = None) -> float:
    """
    Evaluate bbox reward by converting citation coordinates to search image coordinates
    and calculating intersection / ground_truth_area.
    
    Args:
        predict_str: The model's response string
        extra_info: Dictionary containing ground_truth_citation and optionally citation_converted
        retrievaled_images: List of retrieved image paths
        crop_history: List of crop history dictionaries with format:
            [{'bbox': [x1,y1,x2,y2], 'image_path': str, 'step': int}, ...]
    
    Returns:
        bbox_reward score (intersection / ground_truth_area) if evaluation is possible, 0.0 otherwise
    """
    # Get ground truth citation
    ground_truth_citation = extra_info.get('ground_truth_citation') or extra_info.get('bounding_box')
    if not ground_truth_citation:
        return 0.0
    
    # Parse ground truth citation
    if isinstance(ground_truth_citation, str):
        try:
            ground_truth_citation = ast.literal_eval(ground_truth_citation)
        except:
            return 0.0
    
    if not isinstance(ground_truth_citation, list) or len(ground_truth_citation) != 4:
        return 0.0
    
    # 优先使用预计算的citation_converted（如果存在）
    converted_citation = extra_info.get('citation_converted')
    
    if converted_citation is not None and isinstance(converted_citation, list) and len(converted_citation) == 4:
        try:
            bbox_reward = calculate_bbox_reward(converted_citation, ground_truth_citation)
            return bbox_reward
        except Exception as e:
            print(f"[Warning] Failed to calculate bbox_reward with precomputed citation_converted: {e}")
            pass
    
    # 如果没有预计算的citation_converted，则重新计算
    # Extract citation from prediction
    citation = get_citation_from_predict_str(predict_str)
    if citation is None:
        return 0.0
    
    # Build crop chain and convert coordinates
    try:
        if retrievaled_images is None:
            retrievaled_images = extra_info.get('retrievaled_images', [])
        if crop_history is None:
            crop_history = extra_info.get('crop_history', [])
        
        crop_chain = build_crop_chain_from_extra_info(retrievaled_images, crop_history)
        
        # Convert citation coordinates to search image coordinates
        converted_citation = convert_crop_coords_to_search_image(citation, crop_chain)
        
        if converted_citation is None or len(converted_citation) != 4:
            return 0.0
        
        # Calculate bbox_reward
        bbox_reward = calculate_bbox_reward(converted_citation, ground_truth_citation)
        return bbox_reward
        
    except Exception as e:
        # Any error, return 0.0
        return 0.0


def evaluate_citation_iou(predict_str: str, extra_info: Dict, 
                          retrievaled_images: List[str] = None,
                          crop_history: List[Dict] = None) -> float:
    """
    Evaluate citation IoU by converting citation coordinates to search image coordinates.
    
    This function prioritizes using precomputed citation_converted from meta_info (if available),
    otherwise uses the coordinate conversion logic from evaluate_inference.py to ensure
    consistency with evaluation-time logic.
    
    Args:
        predict_str: The model's response string
        extra_info: Dictionary containing ground_truth_citation and optionally citation_converted
        retrievaled_images: List of retrieved image paths
        crop_history: List of crop history dictionaries with format:
            [{'bbox': [x1,y1,x2,y2], 'image_path': str, 'step': int}, ...]
    
    Returns:
        IoU score if evaluation is possible, 0.0 otherwise (consistent with evaluate_inference.py)
    """
    # Get ground truth citation
    ground_truth_citation = extra_info.get('ground_truth_citation') or extra_info.get('bounding_box')
    if not ground_truth_citation:
        return 0.0
    
    # Parse ground truth citation
    if isinstance(ground_truth_citation, str):
        try:
            ground_truth_citation = ast.literal_eval(ground_truth_citation)
        except:
            return 0.0
    
    if not isinstance(ground_truth_citation, list) or len(ground_truth_citation) != 4:
        return 0.0
    
    # 优先使用预计算的citation_converted（如果存在）
    # 这通常来自generation.py中保存的meta_info，对应最后一次迭代的citation
    converted_citation = extra_info.get('citation_converted')
    
    if converted_citation is not None and isinstance(converted_citation, list) and len(converted_citation) == 4:
        # 直接使用预计算的转换坐标（不需要从predict_str中提取citation进行验证）
        # 因为citation_converted是训练时从最后一次迭代的响应中提取并转换的
        try:
            iou = calculate_iou(converted_citation, ground_truth_citation)
            return iou
        except Exception as e:
            # 如果计算失败，继续使用重新计算的方式
            print(f"[Warning] Failed to calculate IoU with precomputed citation_converted: {e}")
            pass
    
    # 如果没有预计算的citation_converted，则重新计算
    # Extract citation from prediction
    citation = get_citation_from_predict_str(predict_str)
    if citation is None:
        return 0.0
    
    # Build crop chain and convert coordinates
    try:
        if retrievaled_images is None:
            retrievaled_images = extra_info.get('retrievaled_images', [])
        if crop_history is None:
            crop_history = extra_info.get('crop_history', [])
        
        crop_chain = build_crop_chain_from_extra_info(retrievaled_images, crop_history)
        
        # Convert citation coordinates to search image coordinates
        converted_citation = convert_crop_coords_to_search_image(citation, crop_chain)
        
        if converted_citation is None or len(converted_citation) != 4:
            return 0.0
        
        # Calculate IoU
        iou = calculate_iou(converted_citation, ground_truth_citation)
        return iou
        
    except Exception as e:
        # Any error, return 0.0 (consistent with evaluate_inference.py)
        return 0.0


def compute_comprehensive_evaluation(predict_str: str, ground_truth: str, extra_info: Dict = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation function that matches evaluate_inference.py logic.
    
    Args:
        predict_str: The model's response string
        ground_truth: The ground truth answer
        extra_info: Dictionary containing:
            - reference_page: List of reference page IDs for search evaluation
            - ground_truth_citation or bounding_box: Ground truth citation coordinates
            - retrievaled_images: List of retrieved image paths (optional)
            - crop_history: List of crop history dictionaries (optional)
    
    Returns:
        Dictionary with evaluation results:
        - search_correct: bool
        - answer_exact_correct: bool
        - answer_containment_correct: bool
        - has_answer: bool
        - has_citation: bool
        - citation_iou: float or None
        - valid_for_evaluation: bool
    """
    if extra_info is None:
        extra_info = {}
    
    result = {
        'search_correct': False,
        'answer_exact_correct': False,
        'answer_containment_correct': False,
        'has_answer': False,
        'has_citation': False,
        'citation_iou': None,
        'valid_for_evaluation': False
    }
    
    # 1. Evaluate search correctness
    retrievaled_images = extra_info.get('retrievaled_images', [])
    result['search_correct'] = evaluate_search_correctness(extra_info, retrievaled_images)
    
    # 2. Evaluate answer correctness
    answer_eval = evaluate_answer_correctness(predict_str, ground_truth)
    result['answer_exact_correct'] = answer_eval['exact_correct']
    result['answer_containment_correct'] = answer_eval['containment_correct']
    result['has_answer'] = answer_eval['has_answer']
    
    # 3. Check for citation
    citation = get_citation_from_predict_str(predict_str)
    result['has_citation'] = citation is not None
    
    # 4. Evaluate citation IoU (only if search is correct and we have all required info)
    if result['search_correct'] and result['has_answer'] and result['has_citation']:
        crop_history = extra_info.get('crop_history', [])
        citation_iou = evaluate_citation_iou(predict_str, extra_info, retrievaled_images, crop_history)
        result['citation_iou'] = citation_iou
        
        # Valid for evaluation if we have all components
        ground_truth_citation = extra_info.get('ground_truth_citation') or extra_info.get('bounding_box')
        if ground_truth_citation and citation_iou is not None:
            result['valid_for_evaluation'] = True
    
    return result

