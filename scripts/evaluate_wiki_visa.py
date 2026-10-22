import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import project_path, record_paths
import re
import ast
import math
import pandas as pd
import importlib.util

# Reuse the exact reward implementation without importing VERL's training runtime.
# Evaluating JSONL on a CPU machine must not require DataProto, Ray or TensorDict.
_metric_path = PROJECT_ROOT / "verl/utils/reward_score/vrag.py"
_metric_spec = importlib.util.spec_from_file_location("mcite_vrag_metrics", _metric_path)
if _metric_spec is None or _metric_spec.loader is None:
    raise ImportError(f"Cannot load reward metrics: {_metric_path}")
vrag = importlib.util.module_from_spec(_metric_spec)
_metric_spec.loader.exec_module(vrag)
normalize_answer = vrag.normalize_answer
get_answer_from_predict_str = vrag.get_answer_from_predict_str
remove_text_between_tags = vrag.remove_text_between_tags
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image


# 缩放参数（与inference_sft_simple.py一致）
MAX_PIXELS = 1536 * 28 * 28  # 401,408
MIN_PIXELS = 768 * 28 * 28  # 200,704

# Default root for resolving wiki-visa reference images
DEFAULT_WIKI_VISA_IMAGE_ROOT = "data/wiki-visa/images"
DEFAULT_WIKI_VISA_PARQUET_ROOT = "data/wiki-visa"

# Binary citation metric threshold: IoU >= threshold => 1 else 0
CITATION_IOU_THRESHOLD = 0.5

# Cache for image index: root -> {stem_or_name: [paths...]}
_IMAGE_INDEX_CACHE: Dict[str, Dict[str, List[str]]] = {}
_BATCH_LAST_SEARCH_Q_CACHE: Dict[str, Dict[str, str]] = {}

def _as_bbox4(obj: Any) -> Optional[List[int]]:
    """
    Normalize bbox-like objects into [x1, y1, x2, y2] as ints.
    Returns None if obj can't be interpreted as a 4-number bbox.
    """
    if obj is None:
        return None
    # Unwrap one level, e.g. [[x1,y1,x2,y2]]
    if isinstance(obj, (list, tuple)) and len(obj) == 1 and isinstance(obj[0], (list, tuple)):
        obj = obj[0]
    if not isinstance(obj, (list, tuple)) or len(obj) != 4:
        return None
    out: List[int] = []
    for v in obj:
        try:
            # Be tolerant to floats/strings
            out.append(int(float(v)))
        except Exception:
            return None
    return out


def _round_answer_stats_for_export(stats: Dict[str, Any], ndigits: int = 2) -> Dict[str, Any]:
    """
    Round answer-related metric floats in stats for easier reading in exported JSON.
    Only touches keys starting with 'answer_' and containing 'accuracy'/'acc'/'rate'.
    """
    out: Dict[str, Any] = dict(stats)
    for k, v in list(out.items()):
        if not isinstance(k, str):
            continue
        if not k.startswith("answer_"):
            continue
        if not any(tok in k for tok in ("accuracy", "acc", "rate")):
            continue
        if isinstance(v, float):
            out[k] = round(v, ndigits)
    return out


def _guess_batch_from_image_path(image_path: Optional[str]) -> Optional[str]:
    """
    Try to infer batch name like 'val_batch_0' or 'train_batch_12' from an image path.
    """
    if not image_path:
        return None
    parts = os.path.normpath(str(image_path)).split(os.sep)
    for p in reversed(parts):
        if re.match(r"^(val|train)_batch_\d+$", p):
            return p
    return None


def _load_last_search_question_map_for_batch(
    batch_name: str,
    parquet_root: str = DEFAULT_WIKI_VISA_PARQUET_ROOT,
) -> Dict[str, str]:
    """
    Build mapping: image_id(stem) -> question for a given batch, using parquet extra_info.
    Cached by parquet path.
    """
    if not batch_name:
        return {}
    parquet_path = os.path.join(parquet_root, f"{batch_name}.parquet")
    if parquet_path in _BATCH_LAST_SEARCH_Q_CACHE:
        return _BATCH_LAST_SEARCH_Q_CACHE[parquet_path]

    mapping: Dict[str, str] = {}
    if not os.path.exists(parquet_path):
        _BATCH_LAST_SEARCH_Q_CACHE[parquet_path] = mapping
        return mapping

    try:
        # Avoid loading huge image bytes; extra_info has image_path + question
        df = pd.read_parquet(parquet_path, columns=["extra_info"])
        for ex in df["extra_info"].tolist():
            if not isinstance(ex, dict):
                continue
            q = ex.get("question")
            img_path = ex.get("image_path") or ex.get("file_name")
            if not q or not img_path:
                continue
            img_id = os.path.splitext(os.path.basename(str(img_path)))[0]
            if img_id and img_id not in mapping:
                mapping[img_id] = str(q)
    except Exception:
        # keep empty mapping on failure
        mapping = {}

    _BATCH_LAST_SEARCH_Q_CACHE[parquet_path] = mapping
    return mapping


def resolve_last_search_question(
    last_search_id: Optional[str],
    last_search_path: Optional[str],
    parquet_root: str = DEFAULT_WIKI_VISA_PARQUET_ROOT,
) -> Optional[str]:
    """
    Resolve last_search_id -> question by loading the corresponding batch parquet.
    """
    if not last_search_id:
        return None
    batch = _guess_batch_from_image_path(last_search_path)
    if not batch:
        return None
    qmap = _load_last_search_question_map_for_batch(batch, parquet_root=parquet_root)
    return qmap.get(last_search_id)


def _build_image_index(image_root: str) -> Dict[str, List[str]]:
    """
    Build an index for fast lookup: map both filename stem and full filename to paths.
    This traverses subdirectories under image_root.
    """
    index: Dict[str, List[str]] = {}
    if not image_root or not os.path.isdir(image_root):
        return index

    for dirpath, _, filenames in os.walk(image_root):
        for fn in filenames:
            # Only index common image extensions
            ext = os.path.splitext(fn)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
                continue
            full_path = os.path.join(dirpath, fn)
            name_key = fn
            stem_key = os.path.splitext(fn)[0]
            index.setdefault(name_key, []).append(full_path)
            index.setdefault(stem_key, []).append(full_path)
    return index


def resolve_reference_image_paths(reference_page: Any, image_root: str = DEFAULT_WIKI_VISA_IMAGE_ROOT) -> List[str]:
    """
    Resolve reference_page IDs/names to actual file paths by traversing image_root subdirs.
    Returns a de-duplicated list of matched paths.
    """
    if not image_root:
        return []

    if image_root not in _IMAGE_INDEX_CACHE:
        _IMAGE_INDEX_CACHE[image_root] = _build_image_index(image_root)

    index = _IMAGE_INDEX_CACHE.get(image_root, {})
    resolved: List[str] = []

    for rid in _reference_ids(reference_page):
        # Try raw and basename variants
        rid_str = str(rid)
        candidates = []
        candidates.extend(index.get(rid_str, []))

        base = os.path.basename(rid_str)
        candidates.extend(index.get(base, []))

        stem = os.path.splitext(base)[0]
        candidates.extend(index.get(stem, []))

        for p in candidates:
            if p not in resolved:
                resolved.append(p)

    return resolved


def process_image_size(original_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    计算图像经过process_image处理后的尺寸（与inference_sft_simple.py的process_image一致）
    
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


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """计算两个bbox的IoU"""
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


def calculate_intersection_over_gt(bbox_pred: List[int], bbox_gt: List[int]) -> float:
    """
    计算交并比中的 intersection / ground_truth_area（类似于召回率）
    """
    if not bbox_pred or not bbox_gt:
        return 0.0

    x1_1, y1_1, x2_1, y2_1 = bbox_pred
    x1_2, y1_2, x2_2, y2_2 = bbox_gt

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    gt_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    if gt_area == 0:
        return 0.0

    return intersection / gt_area


def extract_bbox_from_response(response: str) -> Optional[List[int]]:
    """
    Extract a bbox from model response text.
    Priority:
      1) <bbox>[x1, y1, x2, y2]</bbox>
      2) <citation>[x1, y1, x2, y2]</citation>
      3) first [x1, y1, x2, y2] after </answer>
      4) first [x1, y1, x2, y2] anywhere in response
    """
    if not response:
        return None

    bbox_match = re.search(r"<bbox>\s*\[([^\]]+)\]\s*</bbox>", response, re.IGNORECASE | re.DOTALL)
    if bbox_match:
        try:
            bbox = ast.literal_eval(f"[{bbox_match.group(1)}]")
            if isinstance(bbox, list) and len(bbox) == 4:
                return bbox
        except Exception:
            pass

    citation_match = re.search(r"<citation>\s*\[([^\]]+)\]\s*</citation>", response, re.IGNORECASE | re.DOTALL)
    if citation_match:
        try:
            bbox = ast.literal_eval(f"[{citation_match.group(1)}]")
            if isinstance(bbox, list) and len(bbox) == 4:
                return bbox
        except Exception:
            pass

    answer_end_match = re.search(r"</answer>", response, re.IGNORECASE)
    if answer_end_match:
        after_answer = response[answer_end_match.end() :]
        coord_match = re.search(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", after_answer)
        if coord_match:
            try:
                return [int(coord_match.group(1)), int(coord_match.group(2)), int(coord_match.group(3)), int(coord_match.group(4))]
            except Exception:
                pass

    all_coords = re.findall(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", response)
    if all_coords:
        try:
            first_coord = all_coords[0]
            return [int(first_coord[0]), int(first_coord[1]), int(first_coord[2]), int(first_coord[3])]
        except Exception:
            pass
    return None


def _last_response_after_search(all_responses: List[Dict], all_images: List[Dict]) -> Optional[str]:
    """
    Return the response string immediately after the LAST <search> action.
    This mirrors the heuristic in evaluate_first_retrieval_bbox, but for last search.
    """
    if not all_responses:
        return None

    last_search_response_idx = None
    for i, resp_info in enumerate(all_responses):
        resp = resp_info.get("response", "")
        if resp and re.search(r"<search>", resp, re.IGNORECASE):
            last_search_response_idx = i

    if last_search_response_idx is not None and last_search_response_idx + 1 < len(all_responses):
        return all_responses[last_search_response_idx + 1].get("response", "")

    # Fallback heuristic: if we have at least 2 responses, often response[1] is after first search
    # but for last-search metric we prefer to not guess too aggressively.
    return None


def convert_bbox_to_last_search_image_coords(bbox: List[int], all_images: List[Dict]) -> Optional[List[int]]:
    """
    Convert a bbox produced right after the LAST search to the LAST search image original coordinates.
    Assumption (same as evaluate_first_retrieval_bbox):
      - bbox is relative to the model-input (processed) size of the search image.
      - we map it back to the original size by proportional scaling.
    """
    if not bbox or not all_images:
        return None

    crop_chain = build_crop_chain(all_images)
    search_steps = [s for s in crop_chain if s.get("type") == "search"]
    if not search_steps:
        return None
    last_search_step = search_steps[-1]
    search_image_path = last_search_step.get("source_image")
    if not search_image_path or not os.path.exists(search_image_path):
        return None

    orig_size = get_image_size(search_image_path)
    if not orig_size:
        return None
    proc_size = process_image_size(orig_size)
    w_orig, h_orig = orig_size
    w_proc, h_proc = proc_size

    return [
        int(bbox[0] * w_orig / w_proc) if w_proc > 0 else int(bbox[0]),
        int(bbox[1] * h_orig / h_proc) if h_proc > 0 else int(bbox[1]),
        int(bbox[2] * w_orig / w_proc) if w_proc > 0 else int(bbox[2]),
        int(bbox[3] * h_orig / h_proc) if h_proc > 0 else int(bbox[3]),
    ]


def _normalize_for_relaxed_match(text: str) -> str:
    """Light normalization for relaxed match: lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def relaxed_em(reference_answer: str, predicted_answer: str, max_len_diff: int = 20) -> bool:
    """
    Relaxed exact match:
    - subsequence/substring relationship between ref and pred (either contains the other)
    - length difference no more than max_len_diff characters
    """
    ref = _normalize_for_relaxed_match(reference_answer)
    pred = _normalize_for_relaxed_match(predicted_answer)
    if not ref or not pred:
        return False
    if abs(len(ref) - len(pred)) > max_len_diff:
        return False
    return (ref in pred) or (pred in ref)


def build_crop_chain(all_images: List[Dict]) -> List[Dict]:
    """
    从all_images列表构建裁剪链

    Args:
        all_images: 推理过程中的图像信息列表

    Returns:
        crop_chain: 裁剪链，包含每一步的源图像、bbox和结果图像
    """
    crop_chain = []

    for img_info in all_images:
        if img_info.get('type') == 'search_result':
            # 检索结果
            crop_chain.append({
                'step': len(crop_chain),
                'type': 'search',
                'source_image': img_info.get('path'),
                'bbox': None,
                'result_image': img_info.get('path')
            })
        elif img_info.get('type') == 'crop_result':
            # 裁剪结果 - 源图像是前一个步骤的结果
            source_image = crop_chain[-1]['result_image'] if crop_chain else None
            crop_chain.append({
                'step': len(crop_chain),
                'type': 'crop',
                'source_image': source_image,
                'bbox': img_info.get('bbox'),
                'result_image': img_info.get('path')
            })

    return crop_chain


def convert_crop_coords_to_search_image(citation_coords: List[int], crop_chain: List[Dict]) -> Optional[List[int]]:
    """
    将citation坐标从裁剪坐标系转换到最后一次检索图像坐标系（原始尺寸）
    
    转换流程（根据inference_vrag_20251129.py的实际流程）：
    1. citation坐标是相对于最后一次裁剪并缩放后的图像（模型看到的图像）
    2. 需要反向转换：裁剪后缩放图像 -> 裁剪后原始图像 -> 上一级缩放图像 -> 上一级原始图像 -> ... -> 检索图像原始尺寸
    
    在inference_vrag_20251129.py中的流程：
    - 检索图像（原始）-> process_image缩放 -> 模型看到缩放后的图像
    - 模型在缩放后的图像上生成bbox（相对于缩放后的图像）
    - 将bbox从缩放后的图像映射回原始图像，在原始图像上裁剪
    - 保存裁剪后的原始图像（未缩放）
    - 裁剪后的图像（原始）-> process_image缩放 -> 模型看到缩放后的裁剪图像
    - citation坐标是相对于最后一次裁剪并缩放后的图像
    
    关键点：保存的裁剪图像是从原始图像裁剪出来的（原始尺寸），不是从缩放后的图像裁剪出来的
    
    Args:
        citation_coords: [x1,y1,x2,y2] 在最后一次裁剪并缩放后的图像上的坐标
        crop_chain: 裁剪链信息
    
    Returns:
        [x1,y1,x2,y2] 在最后一次检索图像原始尺寸上的坐标，如果转换失败返回None
    """
    citation_coords = _as_bbox4(citation_coords)
    if not citation_coords:
        return None
    if not crop_chain:
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

    # 当前坐标（相对于最后一次裁剪并缩放后的图像）
    current_coords = citation_coords.copy()

    # 从最后一次裁剪开始，逐步反向转换到检索图像
    for step_idx, step in enumerate(crop_steps_after_search):
        step_bbox = _as_bbox4(step.get('bbox'))
        if not step_bbox:
            continue
        
        # 步骤1: 将坐标从裁剪后缩放图像（模型看到的）转换到裁剪后原始图像（保存的图像）
        # 关键修正：保存的裁剪图像是从原始图像裁剪出来的（原始尺寸），不是从缩放后的图像裁剪出来的
        # 与推理代码一致：inference_vrag_20251129.py中保存的是从raw_image裁剪出来的图像（原始尺寸）
        cropped_image_path = step.get('result_image')
        if not cropped_image_path or not os.path.exists(cropped_image_path):
            return None
        
        # 获取保存的裁剪图像尺寸（这是从原始图像裁剪出来的，原始尺寸）
        cropped_saved_size = get_image_size(cropped_image_path)
        if not cropped_saved_size:
            return None
        
        # 保存的图像是从原始图像裁剪出来的，所以：
        # - cropped_saved_size = 保存的图像尺寸（从原始图像裁剪出来的，原始尺寸）
        # - 这个图像经过process_image处理，得到模型看到的尺寸
        cropped_processed_size = process_image_size(cropped_saved_size)
        cropped_w_saved, cropped_h_saved = cropped_saved_size
        cropped_w_proc, cropped_h_proc = cropped_processed_size
        
        # 计算缩放比例（从模型看到的缩放后图像到保存的裁剪图像）
        # 注意：保存的图像经过process_image处理时，如果像素数不在[MIN_PIXELS, MAX_PIXELS]范围内，会被缩放
        scale_x = cropped_w_saved / cropped_w_proc if cropped_w_proc > 0 else 1.0
        scale_y = cropped_h_saved / cropped_h_proc if cropped_h_proc > 0 else 1.0
        
        # 将坐标从模型看到的缩放后图像转换到保存的裁剪图像（原始尺寸）
        if len(current_coords) != 4:
            return None
        current_coords = [
            int(current_coords[0] * scale_x),
            int(current_coords[1] * scale_y),
            int(current_coords[2] * scale_x),
            int(current_coords[3] * scale_y)
        ]
        
        # 步骤2: 将坐标从裁剪区域转换到上一级图像
        # bbox坐标是在上一级缩放后的图像上的坐标（模型看到的图像）
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = step_bbox
        
        # 需要找到上一级图像，将bbox从缩放后的图像转换到原始图像，然后加上偏移
        if step_idx < len(crop_steps_after_search) - 1:
            # 上一级是前一个裁剪步骤的结果图像
            prev_step = crop_steps_after_search[step_idx + 1]
            prev_image_path = prev_step.get('result_image')
            if prev_image_path and os.path.exists(prev_image_path):
                # 获取上一级保存的裁剪图像尺寸（原始尺寸）
                prev_saved_size = get_image_size(prev_image_path)
                if prev_saved_size:
                    # 上一级保存的图像经过process_image处理，得到模型看到的尺寸
                    prev_processed_size = process_image_size(prev_saved_size)
                    prev_w_saved, prev_h_saved = prev_saved_size
                    prev_w_proc, prev_h_proc = prev_processed_size
                    
                    # 将bbox从上一级缩放后的图像转换到上一级原始图像
                    prev_scale_x = prev_w_saved / prev_w_proc if prev_w_proc > 0 else 1.0
                    prev_scale_y = prev_h_saved / prev_h_proc if prev_h_proc > 0 else 1.0
                    
                    bbox_orig_x1 = int(bbox_x1 * prev_scale_x)
                    bbox_orig_y1 = int(bbox_y1 * prev_scale_y)
                    
                    # 当前坐标是相对于裁剪区域左上角的，需要加上bbox在原始图像上的偏移
                    if len(current_coords) != 4:
                        return None
                    current_coords = [
                        bbox_orig_x1 + current_coords[0],
                        bbox_orig_y1 + current_coords[1],
                        bbox_orig_x1 + current_coords[2],
                        bbox_orig_y1 + current_coords[3]
                    ]
        else:
            # 上一级是检索图像
            search_image_path = last_search_step.get('source_image')
            if search_image_path and os.path.exists(search_image_path):
                search_original_size = get_image_size(search_image_path)
                if search_original_size:
                    search_processed_size = process_image_size(search_original_size)
                    search_w_orig, search_h_orig = search_original_size
                    search_w_proc, search_h_proc = search_processed_size
                    
                    # 将bbox从检索图像的缩放后图像转换到检索图像的原始图像
                    search_scale_x = search_w_orig / search_w_proc if search_w_proc > 0 else 1.0
                    search_scale_y = search_h_orig / search_h_proc if search_h_proc > 0 else 1.0
                    
                    bbox_orig_x1 = int(bbox_x1 * search_scale_x)
                    bbox_orig_y1 = int(bbox_y1 * search_scale_y)
                    
                    # 当前坐标是相对于裁剪区域左上角的，需要加上bbox在原始图像上的偏移
                    if len(current_coords) != 4:
                        return None
                    current_coords = [
                        bbox_orig_x1 + current_coords[0],
                        bbox_orig_y1 + current_coords[1],
                        bbox_orig_x1 + current_coords[2],
                        bbox_orig_y1 + current_coords[3]
                    ]
    
    # 如果没有裁剪步骤，citation坐标直接相对于检索图像的缩放后图像
    if not crop_steps_after_search:
        search_image_path = last_search_step.get('source_image')
        if search_image_path and os.path.exists(search_image_path):
            search_original_size = get_image_size(search_image_path)
            if search_original_size:
                search_processed_size = process_image_size(search_original_size)
                search_w_orig, search_h_orig = search_original_size
                search_w_proc, search_h_proc = search_processed_size
                
                search_scale_x = search_w_orig / search_w_proc if search_w_proc > 0 else 1.0
                search_scale_y = search_h_orig / search_h_proc if search_h_proc > 0 else 1.0
                
                if len(current_coords) != 4:
                    return None
                current_coords = [
                    int(current_coords[0] * search_scale_x),
                    int(current_coords[1] * search_scale_y),
                    int(current_coords[2] * search_scale_x),
                    int(current_coords[3] * search_scale_y)
                ]
    
    return current_coords


def _reference_ids(reference_page: Any) -> List[str]:
    if not reference_page:
        return []

    if hasattr(reference_page, "tolist"):
        page_list = reference_page.tolist()
    elif isinstance(reference_page, list):
        page_list = reference_page
    else:
        page_list = [reference_page]

    return [str(page) for page in page_list if page is not None]


def _last_search_result_id(all_images: List[Dict[str, Any]]) -> Optional[str]:
    for entry in reversed(all_images):
        if entry.get("type") == "search_result":
            path = entry.get("path")
            if path:
                return os.path.splitext(os.path.basename(path.rstrip("/")))[0]
    return None


def _last_search_result_path(all_images: List[Dict[str, Any]]) -> Optional[str]:
    for entry in reversed(all_images):
        if entry.get("type") == "search_result":
            path = entry.get("path")
            if path:
                return path
    return None


def evaluate_sample(
    sample: Dict,
    reference_image_root: Optional[str] = None,
    wiki_visa_parquet_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    评测单个样本

    Args:
        sample: 单个推理结果样本

    Returns:
        评测结果字典
    """
    sample = record_paths(sample, relative=False)
    reference_image_root = str(project_path(reference_image_root)) if reference_image_root else None
    wiki_visa_parquet_root = str(project_path(wiki_visa_parquet_root)) if wiki_visa_parquet_root else None
    uid = sample.get('uid', 'unknown')

    # 3. 获取所有检索到的图像列表 (retrievaled_images)
    # This evaluator primarily expects `all_images` with entries like {"type":"search_result","path":...}.
    # However, some inference pipelines (e.g. vanilla RAG) may output:
    #   - retrieved_images: [ {type:"search_result", path:...}, ... ] OR ["data/wiki-visa/images/page.jpg", ...]
    #   - retrieved_image_path: "data/wiki-visa/images/page.jpg"
    #   - first_retrieved_image: {"path": "..."} (best effort)
    # We add a fallback so the evaluator can still score search/citation when `all_images` is missing.
    all_images = sample.get('all_images', []) or []
    retrievaled_images: List[str] = []
    for img in all_images:
        if not isinstance(img, dict):
            continue
        if img.get('type') == 'search_result':
            path = img.get('path')
            if path:
                retrievaled_images.append(path)

    # Fallback: use top-level fields when `all_images` doesn't include search_result
    if not retrievaled_images:
        ri = sample.get("retrieved_images")
        if isinstance(ri, list):
            for x in ri:
                if isinstance(x, str) and x:
                    retrievaled_images.append(x)
                elif isinstance(x, dict):
                    p = x.get("path") or x.get("source_path")
                    if p:
                        retrievaled_images.append(p)

        rp = sample.get("retrieved_image_path")
        if isinstance(rp, str) and rp:
            retrievaled_images.append(rp)

        fr = sample.get("first_retrieved_image")
        if isinstance(fr, dict):
            p = fr.get("path") or fr.get("source_path")
            if p:
                retrievaled_images.append(p)

    # If we still have no `all_images` for coordinate conversion, synthesize one search_result step
    # so downstream conversion logic can work for single-retrieval pipelines.
    if not any(isinstance(x, dict) and x.get("type") == "search_result" for x in all_images):
        if retrievaled_images:
            all_images = list(all_images) + [{"type": "search_result", "path": retrievaled_images[-1]}]

    # 4. 使用 vrag.evaluate_search_correctness 进行评测
    # 构造符合 vrag 接口的 extra_info
    # vrag 期望 extra_info 包含 'reference_page' (list/str/ndarray) 和 'file_name' (optional)
    reference_page = sample.get('reference_page', [])
    eval_extra_info = {
        'reference_page': reference_page
    }
    search_correct = vrag.evaluate_search_correctness(eval_extra_info, retrievaled_images)
    search_input_correct = False
    reference_ids = _reference_ids(reference_page)
    search_input_id = _last_search_result_id(all_images)
    search_input_path = _last_search_result_path(all_images)
    last_search_question = resolve_last_search_question(
        search_input_id,
        search_input_path,
        parquet_root=wiki_visa_parquet_root or DEFAULT_WIKI_VISA_PARQUET_ROOT,
    )
    if reference_ids and search_input_id:
        search_input_correct = search_input_id in reference_ids
    # print(f"Search correct: {search_correct}, search_input_correct: {search_input_correct}")

    # 获取参考答案和citation
    reference_answer = sample.get('reference_answer', sample.get('ground_truth_answer', ''))
    reference_citation = sample.get('ground_truth_citation', None)
    
    # Fallback: 兼容旧格式，从 bounding_box 字段读取
    if reference_citation is None:
        reference_citation = sample.get('bounding_box', None)
    
    resolved_reference_image_paths = resolve_reference_image_paths(
        reference_page,
        image_root=reference_image_root or DEFAULT_WIKI_VISA_IMAGE_ROOT,
    )

    # 解析标准citation
    if isinstance(reference_citation, str):
        try:
            reference_citation = ast.literal_eval(reference_citation)
        except:
            reference_citation = None

    # 解析生成的答案和citation
    final_answer = None
    final_citation = None
    final_response_raw = None

    all_responses = sample.get('all_responses', [])
    for response_info in reversed(all_responses):
        response = response_info.get('response', '')
        # 与 vrag.compute_answer_acc 对齐：移除用户段后，从最后一个 </answer> 回溯提取 <answer>
        cleaned = remove_text_between_tags(response)
        ans = get_answer_from_predict_str(cleaned)
        if ans is not None:
            final_answer = ans.strip()
            final_response_raw = response
            # 提取 citation（使用多种方法，提高健壮性）
            # 方法1: 标准格式 <citation>[x1, y1, x2, y2]</citation>（大小写不敏感，允许换行）
            citation_match = re.search(r'<citation>\s*\[([^\]]+)\]\s*</citation>', response, re.IGNORECASE | re.DOTALL)
            if citation_match:
                try:
                    parsed = _as_bbox4(ast.literal_eval(f"[{citation_match.group(1)}]"))
                    if parsed is not None:
                        final_citation = parsed
                        break  # 成功提取，退出循环
                    final_citation = None
                except:
                    final_citation = None
            
            # 方法2: 查找 </answer> 后面的坐标（可能没有citation标签）
            answer_end_match = re.search(r'</answer>', response, re.IGNORECASE)
            if answer_end_match:
                after_answer = response[answer_end_match.end():]
                # 查找第一个 [x1, y1, x2, y2] 格式的坐标
                coord_match = re.search(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', after_answer)
                if coord_match:
                    try:
                        final_citation = _as_bbox4([
                            coord_match.group(1), coord_match.group(2),
                            coord_match.group(3), coord_match.group(4),
                        ])
                        break  # 成功提取，退出循环
                    except:
                        final_citation = None
            
            # 方法3: 在response末尾查找坐标（通常citation在最后）
            # 查找最后几个 [x1, y1, x2, y2] 格式的坐标
            all_coords = re.findall(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', response)
            if all_coords:
                try:
                    # 取最后一个坐标（通常citation在最后）
                    last_coord = all_coords[-1]
                    final_citation = _as_bbox4([last_coord[0], last_coord[1], last_coord[2], last_coord[3]])
                    break  # 成功提取，退出循环
                except:
                    final_citation = None
            
            break  # 即使没找到citation，也退出循环（因为已经找到answer了）

    # Fallback: some pipelines may store parsed citation directly
    if final_citation is None:
        c = sample.get("citation")
        if isinstance(c, str):
            try:
                c = ast.literal_eval(c)
            except Exception:
                c = None
        c_norm = _as_bbox4(c)
        if c_norm is not None:
            final_citation = c_norm

    # Fallback: some pipelines may store the final answer directly
    if final_answer is None:
        a = sample.get("answer")
        if isinstance(a, str) and a.strip():
            cleaned = remove_text_between_tags(a)
            ans = get_answer_from_predict_str(cleaned)
            final_answer = (ans.strip() if ans is not None else a.strip())
            final_response_raw = a

    # 使用 vrag.evaluate_answer_correctness 复核答案准确率
    # 注意：我们需要传入包含 <answer> 标签的原始预测字符串给 vrag 函数，因为它内部会再次提取
    # 但我们已经手动提取了 final_answer，为了复用逻辑，我们构造一个模拟的 predict_str
    # 或者，如果 final_answer 已经提取出来了，我们可以直接使用 normalize_answer 进行比较，这正是 evaluate_inference.py 已经做的

    # 初始化评测结果
    result = {
        'uid': uid,
        # debug fields (help analyze cases where answer is correct but search is not)
        'query': sample.get('query', ''),
        'reference_page': reference_page,
        'reference_image_paths': resolved_reference_image_paths,
        'retrievaled_images': retrievaled_images,
        'debug_last_search_id': search_input_id,
        'debug_last_search_path': search_input_path,
        'debug_last_search_question': last_search_question,
        'reference_answer': reference_answer,
        'final_answer': final_answer,
        'final_response_raw': final_response_raw,
        'search_correct': search_correct,
        'search_input_correct': search_input_correct,
        'has_answer': final_answer is not None,
        'has_citation': final_citation is not None,
        'answer_exact_correct': False,  # 精确匹配
        'answer_containment_correct': False,  # 包含匹配
        'answer_relaxed_em_correct': False,  # relaxed EM (subsequence + <=20 chars diff)
        'answer_correct': False,  # 只在检索正确时计算
        'answer_exact_correct_when_search_input_correct': False,  # 只在检索对正确时计算的精确匹配
        'answer_containment_correct_when_search_input_correct': False,  # 只在检索对正确时计算的包含匹配
        'answer_relaxed_em_correct_when_search_input_correct': False,  # relaxed EM when search input correct
        'citation_iou': None,     # 只在检索正确时计算
        'citation_intersection_over_gt': None,  # intersection / ground_truth_area
        # Binary citation metric: 1 if citation_iou >= threshold else 0 (only meaningful when citation_iou is not None)
        'citation_iou_ge_threshold': None,
        'citation_iou_threshold': CITATION_IOU_THRESHOLD,
        # first bbox metric after last search (only meaningful when last search input is correct)
        'first_bbox_after_last_search': None,  # raw bbox extracted from response (processed coords)
        'first_bbox_after_last_search_converted': None,  # converted to last search original coords
        'first_bbox_after_last_search_intersection_over_gt': None,  # intersection / gt_area
        'first_bbox_after_last_search_failure_reason': None,
        'valid_for_evaluation': False
    }

    # 答案EM（严格对齐 vrag.compute_answer_acc，全集口径，没答出即错误）
    # 这里的逻辑与 vrag.compute_answer_acc 基本一致：
    # 1. vrag.compute_answer_acc: remove_text_between_tags -> get_answer_from_predict_str -> normalize -> compare
    # 2. 这里: remove_text_between_tags -> get_answer_from_predict_str (上面已做) -> normalize -> compare
    if reference_answer:
        if final_answer is not None:
            ref_norm = normalize_answer(reference_answer)
            pred_norm = normalize_answer(final_answer)
            result['answer_exact_correct'] = (ref_norm == pred_norm)
            # 兼容原有包含匹配统计（轻量规范化）
            # vrag.evaluate_answer_correctness 也是这样做的
            ref_clean = re.sub(r'\s+', ' ', reference_answer.lower().strip())
            pred_clean = re.sub(r'\s+', ' ', final_answer.lower().strip())
            result['answer_containment_correct'] = ref_clean in pred_clean
            result['answer_relaxed_em_correct'] = relaxed_em(reference_answer, final_answer, max_len_diff=20)
        else:
            result['answer_exact_correct'] = False
            result['answer_containment_correct'] = False
            result['answer_relaxed_em_correct'] = False
    
    # 只在检索对正确的情况下才计算答案指标
    if search_input_correct:
        result['answer_exact_correct_when_search_input_correct'] = result['answer_exact_correct']
        result['answer_containment_correct_when_search_input_correct'] = result['answer_containment_correct']
        result['answer_relaxed_em_correct_when_search_input_correct'] = result['answer_relaxed_em_correct']

    # ---- New metric: first bbox after LAST search, intersection / GT_area (when last search input correct) ----
    # This follows evaluate_first_retrieval_bbox logic:
    # - take response right after the last <search>
    # - extract <bbox> (or fallback) from that response
    # - convert bbox from processed coords -> original coords of last search image
    # - compute intersection / ground_truth_area vs reference_citation
    if search_input_correct:
        if not reference_citation or not isinstance(reference_citation, list) or len(reference_citation) != 4:
            result['first_bbox_after_last_search_failure_reason'] = 'no_reference_citation'
            result['first_bbox_after_last_search_intersection_over_gt'] = None
        else:
            resp_after_search = _last_response_after_search(all_responses, all_images)
            if not resp_after_search:
                result['first_bbox_after_last_search_failure_reason'] = 'no_response_after_last_search'
                # keep None to avoid mixing "no signal" with "0 overlap"
                result['first_bbox_after_last_search_intersection_over_gt'] = None
            else:
                first_bbox = extract_bbox_from_response(resp_after_search)
                result['first_bbox_after_last_search'] = first_bbox
                if not first_bbox:
                    result['first_bbox_after_last_search_failure_reason'] = 'no_bbox_in_response_after_last_search'
                    # When last search is correct, absence of bbox is treated as 0 overlap
                    result['first_bbox_after_last_search_intersection_over_gt'] = 0.0
                else:
                    converted = convert_bbox_to_last_search_image_coords(first_bbox, all_images)
                    result['first_bbox_after_last_search_converted'] = converted
                    if not converted:
                        result['first_bbox_after_last_search_failure_reason'] = 'bbox_coordinate_conversion_failed'
                        # Treat as 0 overlap (consistent with missing bbox handling)
                        result['first_bbox_after_last_search_intersection_over_gt'] = 0.0
                    else:
                        result['first_bbox_after_last_search_intersection_over_gt'] = calculate_intersection_over_gt(
                            converted, reference_citation
                        )

    # 只要检索正确，就计算citation IoU
    # 没有生成citation或答案时，IoU设为0.0
    valid_for_evaluation = search_correct
    result['conversion_failure_reason'] = None

    if not valid_for_evaluation:
        result['citation_iou'] = None
        result['citation_intersection_over_gt'] = None
        result['citation_iou_ge_threshold'] = None
        result['valid_for_evaluation'] = False
        if not search_correct:
            result['conversion_failure_reason'] = 'search_incorrect'
        return result

    result['valid_for_evaluation'] = True

    if not reference_citation:
        result['citation_iou'] = 0.0
        result['citation_intersection_over_gt'] = 0.0
        result['citation_iou_ge_threshold'] = 1 if result['citation_iou'] >= CITATION_IOU_THRESHOLD else 0
        result['conversion_failure_reason'] = 'no_reference_citation'
        # print(f"[{uid}] Warning: No reference citation, setting IoU to 0.0")
        return result

    if not final_citation:
        result['citation_iou'] = 0.0
        result['citation_intersection_over_gt'] = 0.0
        result['citation_iou_ge_threshold'] = 1 if result['citation_iou'] >= CITATION_IOU_THRESHOLD else 0
        result['conversion_failure_reason'] = 'no_generated_citation'
        # print(f"[{uid}] Warning: No generated citation, setting IoU to 0.0")
        return result

    converted_citation = sample.get('citation_converted')
    if converted_citation is not None and isinstance(converted_citation, list) and len(converted_citation) == 4:
        if len(reference_citation) == 4:
            citation_iou = calculate_iou(converted_citation, reference_citation)
            result['citation_iou'] = citation_iou
            result['citation_intersection_over_gt'] = calculate_intersection_over_gt(
                converted_citation, reference_citation
            )
            result['citation_iou_ge_threshold'] = 1 if citation_iou >= CITATION_IOU_THRESHOLD else 0
            result['debug_converted_coords'] = converted_citation
            result['debug_original_coords'] = final_citation
            result['conversion_method'] = 'use_precomputed'
        else:
            result['citation_iou'] = 0.0
            result['citation_intersection_over_gt'] = 0.0
            result['citation_iou_ge_threshold'] = 1 if result['citation_iou'] >= CITATION_IOU_THRESHOLD else 0
            result['conversion_failure_reason'] = 'invalid_reference_citation_format'
            # print(f"[{uid}] Warning: Invalid reference citation format: {reference_citation}")
        return result

    result['conversion_method'] = 'recompute'
    crop_chain = build_crop_chain(all_images)
    final_citation_norm = _as_bbox4(final_citation)
    converted_citation = convert_crop_coords_to_search_image(final_citation_norm, crop_chain)

    # Extra fallback: if there is no crop/search chain info (or chain is empty) but we have a retrieved image path,
    # treat `final_citation` as relative to the processed (model-input) retrieved image and scale back to original.
    if (not crop_chain) and final_citation_norm and retrievaled_images:
        last_img = retrievaled_images[-1]
        if last_img and os.path.exists(last_img):
            orig_size = get_image_size(last_img)
            if orig_size:
                proc_size = process_image_size(orig_size)
                w_orig, h_orig = orig_size
                w_proc, h_proc = proc_size
                converted_citation = [
                    int(final_citation_norm[0] * w_orig / w_proc) if w_proc > 0 else int(final_citation_norm[0]),
                    int(final_citation_norm[1] * h_orig / h_proc) if h_proc > 0 else int(final_citation_norm[1]),
                    int(final_citation_norm[2] * w_orig / w_proc) if w_proc > 0 else int(final_citation_norm[2]),
                    int(final_citation_norm[3] * h_orig / h_proc) if h_proc > 0 else int(final_citation_norm[3]),
                ]

    if converted_citation is not None and len(reference_citation) == 4:
        citation_iou = calculate_iou(converted_citation, reference_citation)
        result['citation_iou'] = citation_iou
        result['citation_intersection_over_gt'] = calculate_intersection_over_gt(
            converted_citation, reference_citation
        )
        result['citation_iou_ge_threshold'] = 1 if citation_iou >= CITATION_IOU_THRESHOLD else 0
        result['debug_converted_coords'] = converted_citation
        result['debug_original_coords'] = final_citation_norm
        # print(f"[{uid}] ⚠️ Recomputed citation conversion (citation_converted not found): {final_citation} -> {converted_citation}")
    else:
        result['citation_iou'] = 0.0
        result['citation_intersection_over_gt'] = 0.0
        result['citation_iou_ge_threshold'] = 1 if result['citation_iou'] >= CITATION_IOU_THRESHOLD else 0
        result['conversion_failure_reason'] = 'coordinate_conversion_failed'
        # print(f"[{uid}] Warning: Failed to convert citation coordinates (no citation_converted and recompute failed)")

    return result


def evaluate_inference_results(
    inference_file: str,
    max_samples: Optional[int] = None,
    reference_image_root: Optional[str] = None,
    wiki_visa_parquet_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    评测推理结果：answer用包含关系，citation用IoU（转换到检索图像坐标系）

    Args:
        inference_file: 推理结果文件路径（jsonl格式）
        max_samples: 最大评估样本数，如果指定则只评估前N个样本

    Returns:
        评测统计结果
    """
    inference_file = str(project_path(inference_file))
    reference_image_root = str(project_path(reference_image_root)) if reference_image_root else None
    wiki_visa_parquet_root = str(project_path(wiki_visa_parquet_root)) if wiki_visa_parquet_root else None
    print(f"Evaluating inference results from {inference_file}...")

    # 读取推理结果
    results = []
    with open(inference_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    results.append(record_paths(json.loads(line), relative=False))
                    # 如果指定了max_samples，达到数量后停止读取
                    if max_samples is not None and len(results) >= max_samples:
                        break
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")

    # 如果指定了max_samples，确保只取前N个
    if max_samples is not None and len(results) > max_samples:
        results = results[:max_samples]
    
    print(f"Loaded {len(results)} inference results" + (f" (limited to first {max_samples} samples)" if max_samples else ""))

    # 评测每个样本
    sample_results = []
    for i, sample in enumerate(results):
        if (i + 1) % 100 == 0:
            print(f"Processing sample {i + 1}/{len(results)}...")

        sample_result = evaluate_sample(
            sample,
            reference_image_root=reference_image_root,
            wiki_visa_parquet_root=wiki_visa_parquet_root,
        )
        sample_results.append(sample_result)

    # 统计结果
    total_samples = len(sample_results)
    search_correct_count = sum(1 for r in sample_results if r.get('search_correct', False))
    search_input_correct_count = sum(1 for r in sample_results if r.get('search_input_correct', False))
    valid_samples = sum(1 for r in sample_results if r['valid_for_evaluation'])  # 检索正确的样本（都会计算IoU）

    # 计数器：这些计数器统计的是完全正确的样本数 (Search Correct + Answer Match + Has Citation)
    answer_exact_correct_count = sum(1 for r in sample_results if r['answer_exact_correct'])
    answer_containment_correct_count = sum(1 for r in sample_results if r['answer_containment_correct'])
    answer_relaxed_em_correct_count = sum(1 for r in sample_results if r.get('answer_relaxed_em_correct', False))
    
    # 统计生成citations的样本数量
    samples_with_citation = sum(1 for r in sample_results if r.get('has_citation', False))

    # citation_iou_scores包含所有检索正确样本的IoU值（没有citation或转换失败时设为0.0）
    # 检索不正确的样本citation_iou为None，不会被包含
    citation_iou_scores = [r['citation_iou'] for r in sample_results if r['citation_iou'] is not None]
    
    # citation_iou_scores_with_citation只包含有citation的样本的IoU值（去掉没有citation的样本）
    # 只统计那些有生成citation且转换成功的样本
    citation_iou_scores_with_citation = [
        r['citation_iou'] for r in sample_results 
        if r['citation_iou'] is not None 
        and r.get('has_citation', False) 
        and r.get('conversion_failure_reason') is None
    ]
    
    # 基于search_input_correct的citation平均IoU（检索正确的里面citation的平均iou）
    citation_iou_scores_search_input_correct = [
        r['citation_iou'] for r in sample_results 
        if r.get('search_input_correct', False) and r['citation_iou'] is not None
    ]

    # Binary citation metric: IoU >= threshold
    citation_iou_ge_threshold_valid = [
        r.get('citation_iou_ge_threshold') for r in sample_results
        if r.get('citation_iou_ge_threshold') is not None
    ]
    citation_iou_ge_threshold_search_input_correct = [
        r.get('citation_iou_ge_threshold') for r in sample_results
        if r.get('search_input_correct', False) and r.get('citation_iou_ge_threshold') is not None
    ]

    # intersection / ground_truth_area 指标
    citation_iog_scores = [
        r['citation_intersection_over_gt'] for r in sample_results
        if r.get('citation_intersection_over_gt') is not None
    ]
    citation_iog_scores_with_citation = [
        r['citation_intersection_over_gt'] for r in sample_results
        if r.get('citation_intersection_over_gt') is not None
        and r.get('has_citation', False)
        and r.get('conversion_failure_reason') is None
    ]
    citation_iog_scores_search_input_correct = [
        r['citation_intersection_over_gt'] for r in sample_results
        if r.get('search_input_correct', False) and r.get('citation_intersection_over_gt') is not None
    ]

    # New metric: first bbox (after last search) intersection / GT_area, averaged over last-search-correct samples
    first_bbox_iog_scores_search_input_correct = [
        r.get('first_bbox_after_last_search_intersection_over_gt')
        for r in sample_results
        if r.get('search_input_correct', False) and r.get('first_bbox_after_last_search_intersection_over_gt') is not None
    ]

    # 统计转换失败的原因
    conversion_failures = {}
    for r in sample_results:
        if r.get('conversion_failure_reason'):
            reason = r['conversion_failure_reason']
            conversion_failures[reason] = conversion_failures.get(reason, 0) + 1
    
    # 收集转换失败的样本uid（用于调试）
    conversion_failure_samples = {}
    for r in sample_results:
        if r.get('conversion_failure_reason'):
            reason = r['conversion_failure_reason']
            if reason not in conversion_failure_samples:
                conversion_failure_samples[reason] = []
            conversion_failure_samples[reason].append(r.get('uid', 'unknown'))

    # 计算统计指标
    answer_exact_search_input_count = sum(1 for r in sample_results if r.get('search_input_correct', False) and r.get('answer_exact_correct', False))
    answer_containment_search_input_count = sum(1 for r in sample_results if r.get('search_input_correct', False) and r.get('answer_containment_correct', False))
    answer_relaxed_em_search_input_count = sum(1 for r in sample_results if r.get('search_input_correct', False) and r.get('answer_relaxed_em_correct', False))

    # 基于 search_correct（只要检索过程中出现过正确图像就算检索成功）的 relaxed EM 正确数量
    answer_relaxed_em_search_correct_count = sum(
        1 for r in sample_results if r.get('search_correct', False) and r.get('answer_relaxed_em_correct', False)
    )

    # search 不正确但 relaxed answer 正确（你看到的“112条”）
    search_incorrect_relaxed_correct = [
        r for r in sample_results
        if (not r.get('search_correct', False)) and r.get('answer_relaxed_em_correct', False)
    ]
    search_incorrect_relaxed_correct_count = len(search_incorrect_relaxed_correct)
    
    # 只在检索对正确的情况下才计算的答案指标
    answer_exact_when_search_input_correct_count = sum(1 for r in sample_results if r.get('answer_exact_correct_when_search_input_correct', False))
    answer_containment_when_search_input_correct_count = sum(1 for r in sample_results if r.get('answer_containment_correct_when_search_input_correct', False))
    answer_relaxed_em_when_search_input_correct_count = sum(1 for r in sample_results if r.get('answer_relaxed_em_correct_when_search_input_correct', False))

    stats = {
        'max_samples': max_samples,  # 保存max_samples信息，用于报告显示
        'total_samples': total_samples,
        'search_correct_count': search_correct_count,
        'search_input_correct_count': search_input_correct_count,
        'search_accuracy': search_correct_count / total_samples if total_samples > 0 else 0.0,
        'search_input_accuracy': search_input_correct_count / total_samples if total_samples > 0 else 0.0,
        'answer_exact_accuracy_search_input': answer_exact_search_input_count / search_input_correct_count if search_input_correct_count > 0 else 0.0,
        'answer_containment_accuracy_search_input': answer_containment_search_input_count / search_input_correct_count if search_input_correct_count > 0 else 0.0,
        'answer_relaxed_em_accuracy_search_input': answer_relaxed_em_search_input_count / search_input_correct_count if search_input_correct_count > 0 else 0.0,
        'answer_exact_search_input_count': answer_exact_search_input_count,  # 保存实际数量
        'answer_containment_search_input_count': answer_containment_search_input_count,  # 保存实际数量
        'answer_relaxed_em_search_input_count': answer_relaxed_em_search_input_count,

        # **新增：search_correct 条件下的 relaxed EM 正确数量/比例**
        'answer_relaxed_em_search_correct_count': answer_relaxed_em_search_correct_count,
        'answer_relaxed_em_accuracy_search_correct': answer_relaxed_em_search_correct_count / search_correct_count if search_correct_count > 0 else 0.0,

        'search_incorrect_relaxed_correct_count': search_incorrect_relaxed_correct_count,
        # 只在检索对正确的情况下才计算的答案指标
        'answer_exact_when_search_input_correct_count': answer_exact_when_search_input_correct_count,
        'answer_containment_when_search_input_correct_count': answer_containment_when_search_input_correct_count,
        'answer_relaxed_em_when_search_input_correct_count': answer_relaxed_em_when_search_input_correct_count,
        'answer_exact_accuracy_when_search_input_correct': answer_exact_when_search_input_correct_count / search_input_correct_count if search_input_correct_count > 0 else 0.0,
        'answer_containment_accuracy_when_search_input_correct': answer_containment_when_search_input_correct_count / search_input_correct_count if search_input_correct_count > 0 else 0.0,
        'answer_relaxed_em_accuracy_when_search_input_correct': answer_relaxed_em_when_search_input_correct_count / search_input_correct_count if search_input_correct_count > 0 else 0.0,
        'valid_samples': valid_samples,

        # **新的指标：在全集上计算准确率**
        'answer_exact_correct_count': answer_exact_correct_count,
        'answer_containment_correct_count': answer_containment_correct_count,
        'answer_relaxed_em_correct_count': answer_relaxed_em_correct_count,
        'answer_exact_accuracy_full_set': answer_exact_correct_count / total_samples if total_samples > 0 else 0.0,
        'answer_containment_accuracy_full_set': answer_containment_correct_count / total_samples if total_samples > 0 else 0.0,
        'answer_relaxed_em_accuracy_full_set': answer_relaxed_em_correct_count / total_samples if total_samples > 0 else 0.0,

        # 保留原有的基于valid_samples的准确率（用于citation IoU的基准）
        'answer_exact_accuracy_valid_set': answer_exact_correct_count / valid_samples if valid_samples > 0 else 0.0,
        'answer_containment_accuracy_valid_set': answer_containment_correct_count / valid_samples if valid_samples > 0 else 0.0,
        'answer_relaxed_em_accuracy_valid_set': answer_relaxed_em_correct_count / valid_samples if valid_samples > 0 else 0.0,
        
        # 生成citations的样本数量
        'samples_with_citation': samples_with_citation,
        'samples_with_citation_rate': samples_with_citation / total_samples if total_samples > 0 else 0.0,

        # IoU指标1：包含所有检索正确样本（没有citation时设为0.0）
        'citation_iou_mean': sum(citation_iou_scores) / len(citation_iou_scores) if citation_iou_scores else 0.0,
        'citation_iou_std': (sum(
            (x - sum(citation_iou_scores) / len(citation_iou_scores)) ** 2 for x in citation_iou_scores) / len(
            citation_iou_scores)) ** 0.5 if citation_iou_scores else 0.0,
        'citation_iou_median': sorted(citation_iou_scores)[
            len(citation_iou_scores) // 2] if citation_iou_scores else 0.0,
        'citation_iou_valid_count': len(citation_iou_scores),
        
        # IoU指标2：只包含有citation的样本（去掉没有citation的）
        'citation_iou_mean_with_citation': sum(citation_iou_scores_with_citation) / len(citation_iou_scores_with_citation) if citation_iou_scores_with_citation else 0.0,
        'citation_iou_std_with_citation': (sum(
            (x - sum(citation_iou_scores_with_citation) / len(citation_iou_scores_with_citation)) ** 2 for x in citation_iou_scores_with_citation) / len(
            citation_iou_scores_with_citation)) ** 0.5 if citation_iou_scores_with_citation else 0.0,
        'citation_iou_median_with_citation': sorted(citation_iou_scores_with_citation)[
            len(citation_iou_scores_with_citation) // 2] if citation_iou_scores_with_citation else 0.0,
        'citation_iou_with_citation_count': len(citation_iou_scores_with_citation),
        
        # IoU指标3：基于search_input_correct的citation平均IoU（检索正确的里面citation的平均iou）
        'citation_iou_mean_search_input_correct': sum(citation_iou_scores_search_input_correct) / len(citation_iou_scores_search_input_correct) if citation_iou_scores_search_input_correct else 0.0,

        # Binary citation metric: IoU >= threshold => 1 else 0
        'citation_iou_threshold': CITATION_IOU_THRESHOLD,
        'citation_iou_ge_threshold_count_valid': sum(citation_iou_ge_threshold_valid) if citation_iou_ge_threshold_valid else 0,
        'citation_iou_ge_threshold_rate_valid': (
            sum(citation_iou_ge_threshold_valid) / len(citation_iou_ge_threshold_valid)
            if citation_iou_ge_threshold_valid
            else 0.0
        ),
        'citation_iou_ge_threshold_count_search_input_correct': (
            sum(citation_iou_ge_threshold_search_input_correct) if citation_iou_ge_threshold_search_input_correct else 0
        ),
        'citation_iou_ge_threshold_rate_search_input_correct': (
            sum(citation_iou_ge_threshold_search_input_correct) / len(citation_iou_ge_threshold_search_input_correct)
            if citation_iou_ge_threshold_search_input_correct
            else 0.0
        ),

        # intersection / ground_truth_area 指标
        'citation_intersection_over_gt_mean': sum(citation_iog_scores) / len(citation_iog_scores) if citation_iog_scores else 0.0,
        'citation_intersection_over_gt_mean_with_citation': sum(citation_iog_scores_with_citation) / len(citation_iog_scores_with_citation) if citation_iog_scores_with_citation else 0.0,
        'citation_intersection_over_gt_mean_search_input_correct': sum(citation_iog_scores_search_input_correct) / len(citation_iog_scores_search_input_correct) if citation_iog_scores_search_input_correct else 0.0,

        # New: first bbox intersection/gt mean (when last search input correct)
        'first_bbox_intersection_over_gt_mean_search_input_correct': (
            sum(first_bbox_iog_scores_search_input_correct) / len(first_bbox_iog_scores_search_input_correct)
            if first_bbox_iog_scores_search_input_correct
            else 0.0
        ),
        'first_bbox_intersection_over_gt_count_search_input_correct': len(first_bbox_iog_scores_search_input_correct),
        
        # 转换失败统计
        'conversion_failures': conversion_failures,
        'conversion_failure_samples': conversion_failure_samples,
        'total_conversion_failures': sum(conversion_failures.values())
    }

    # IoU分布统计（基于所有样本）
    if citation_iou_scores:
        iou_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        stats['iou_distribution'] = {}
        for min_iou, max_iou in iou_ranges:
            count = sum(1 for iou in citation_iou_scores if min_iou <= iou < max_iou)
            percentage = count / len(citation_iou_scores) * 100
            stats['iou_distribution'][f'{min_iou}-{max_iou}'] = {
                'count': count,
                'percentage': percentage
            }
        # IoU >= 0.9 的样本
        high_iou_count = sum(1 for iou in citation_iou_scores if iou >= 0.9)
        stats['iou_distribution']['>=0.9'] = {
            'count': high_iou_count,
            'percentage': high_iou_count / len(citation_iou_scores) * 100
        }
    
    # IoU分布统计（只基于有citation的样本）
    if citation_iou_scores_with_citation:
        iou_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        stats['iou_distribution_with_citation'] = {}
        for min_iou, max_iou in iou_ranges:
            count = sum(1 for iou in citation_iou_scores_with_citation if min_iou <= iou < max_iou)
            percentage = count / len(citation_iou_scores_with_citation) * 100
            stats['iou_distribution_with_citation'][f'{min_iou}-{max_iou}'] = {
                'count': count,
                'percentage': percentage
            }
        # IoU >= 0.9 的样本
        high_iou_count = sum(1 for iou in citation_iou_scores_with_citation if iou >= 0.9)
        stats['iou_distribution_with_citation']['>=0.9'] = {
            'count': high_iou_count,
            'percentage': high_iou_count / len(citation_iou_scores_with_citation) * 100
        }


    return stats, sample_results



def print_evaluation_report(
    stats: Dict[str, Any],
    sample_results: List[Dict[str, Any]],
    show_search_incorrect_relaxed_correct: int = 10,
    dump_search_incorrect_relaxed_correct: Optional[str] = None,
    print_model_output: bool = True,
    max_model_output_chars: int = 800,
    export_last_search_examples: Optional[str] = None,
    export_last_search_examples_all: bool = False,
):
    """打印评测报告"""
    print("\n" + "=" * 70)
    print("INFERENCE EVALUATION REPORT")
    print("=" * 70)

    print(f"Total samples: {stats['total_samples']}")
    
    search_acc = stats['search_input_correct_count'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0.0
    print(f"Search correct count (Last search input correct): {stats['search_input_correct_count']} ({search_acc:.1f}%)")

    search_any_acc = stats.get('search_correct_count', 0) / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0.0
    print(f"Search correct count (Any retrieved image correct): {stats.get('search_correct_count', 0)} ({search_any_acc:.1f}%)")
    
    answer_acc_total = stats['answer_exact_correct_count'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0.0
    print(f"Answer exact correct count (Total): {stats['answer_exact_correct_count']} ({answer_acc_total:.2f}%)")
    
    answer_cont_total = stats['answer_containment_correct_count'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0.0
    print(f"Answer containment correct count (Total): {stats['answer_containment_correct_count']} ({answer_cont_total:.2f}%)")
    
    answer_relaxed_total = stats.get('answer_relaxed_em_correct_count', 0) / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0.0
    print(f"Answer relaxed EM correct count (Total): {stats.get('answer_relaxed_em_correct_count', 0)} ({answer_relaxed_total:.2f}%)")

    relaxed_any_acc = stats.get('answer_relaxed_em_search_correct_count', 0) / stats.get('search_correct_count', 0) * 100 if stats.get('search_correct_count', 0) > 0 else 0.0
    print(
        "Answer relaxed EM correct count (When any retrieved image correct): "
        f"{stats.get('answer_relaxed_em_search_correct_count', 0)} ({relaxed_any_acc:.2f}%)"
    )
    print(
        "Search incorrect but relaxed EM correct count: "
        f"{stats.get('search_incorrect_relaxed_correct_count', 0)}"
    )

    answer_acc_conditional = stats['answer_exact_when_search_input_correct_count'] / stats['search_input_correct_count'] * 100 if stats['search_input_correct_count'] > 0 else 0.0
    print(f"Answer exact correct count (When search input correct): {stats['answer_exact_when_search_input_correct_count']} ({answer_acc_conditional:.2f}%)")
    
    answer_cont_conditional = stats['answer_containment_when_search_input_correct_count'] / stats['search_input_correct_count'] * 100 if stats['search_input_correct_count'] > 0 else 0.0
    print(f"Answer containment correct count (When search input correct): {stats['answer_containment_when_search_input_correct_count']} ({answer_cont_conditional:.2f}%)")
    
    answer_relaxed_conditional = stats.get('answer_relaxed_em_when_search_input_correct_count', 0) / stats['search_input_correct_count'] * 100 if stats['search_input_correct_count'] > 0 else 0.0
    print(f"Answer relaxed EM correct count (When search input correct): {stats.get('answer_relaxed_em_when_search_input_correct_count', 0)} ({answer_relaxed_conditional:.2f}%)")
    
    print(f"Citation Mean IoU (When search input correct): {stats['citation_iou_mean_search_input_correct']:.4f}")
    print(f"Citation Mean Intersection/GroundTruth (When search input correct): {stats['citation_intersection_over_gt_mean_search_input_correct']:.4f}")
    thr = stats.get('citation_iou_threshold', CITATION_IOU_THRESHOLD)
    print(
        f"Citation IoU>={thr} rate (Valid set): "
        f"{stats.get('citation_iou_ge_threshold_count_valid', 0)}/{stats.get('citation_iou_valid_count', 0)} "
        f"({stats.get('citation_iou_ge_threshold_rate_valid', 0.0) * 100:.1f}%)"
    )
    print(
        f"Citation IoU>={thr} rate (When search input correct): "
        f"{stats.get('citation_iou_ge_threshold_count_search_input_correct', 0)}/{stats.get('search_input_correct_count', 0)} "
        f"({stats.get('citation_iou_ge_threshold_rate_search_input_correct', 0.0) * 100:.1f}%)"
    )
    print(
        "First BBox Mean Intersection/GroundTruth (After last search, when search input correct): "
        f"{stats.get('first_bbox_intersection_over_gt_mean_search_input_correct', 0.0):.4f} "
        f"(n={stats.get('first_bbox_intersection_over_gt_count_search_input_correct', 0)})"
    )

    # NOTE: Example printing / exporting logic is intentionally disabled (not needed for current runs).
    # If you want it back, revert this block and re-enable the related args wiring.
    #
    # # Inspect "the 112" cases: search incorrect but relaxed answer correct
    # cases = [
    #     r for r in sample_results
    #     if (not r.get('search_correct', False)) and r.get('answer_relaxed_em_correct', False)
    # ]
    # export_cases = cases if export_last_search_examples_all else cases[:show_search_incorrect_relaxed_correct]
    # if export_last_search_examples and export_cases:
    #     ...
    # if dump_search_incorrect_relaxed_correct:
    #     ...
    # if show_search_incorrect_relaxed_correct and show_search_incorrect_relaxed_correct > 0 and cases:
    #     ...

    print("=" * 70)



def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate inference results')
    parser.add_argument('--inference_file', required=True, help='Path to inference results file (jsonl format)')
    parser.add_argument('--output_file', help='Path to save detailed results (optional)')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to evaluate (evaluate first N samples only)')
    parser.add_argument('--show_search_incorrect_relaxed_correct', type=int, default=10, help='Print N examples where search is incorrect but relaxed EM is correct')
    parser.add_argument('--dump_search_incorrect_relaxed_correct', default=None, help='Dump all such cases to a jsonl file')
    parser.add_argument('--print_model_output', action='store_true', help='Print raw model output for the shown examples')
    parser.add_argument('--max_model_output_chars', type=int, default=800, help='Max chars of model output to print per example')
    parser.add_argument('--export_last_search_examples', default=None, help='Export shown examples to jsonl with last_search_id/path and full data')
    parser.add_argument('--export_last_search_examples_all', action='store_true', help='Export ALL such examples (not only shown ones)')
    parser.add_argument('--reference_image_root', default=DEFAULT_WIKI_VISA_IMAGE_ROOT, help='Root directory to resolve reference_page image paths (will traverse subdirs)')
    parser.add_argument('--wiki_visa_parquet_root', default=DEFAULT_WIKI_VISA_PARQUET_ROOT, help='Root directory containing wiki-visa batch parquet files')

    args = parser.parse_args()
    for name in ("inference_file", "output_file", "reference_image_root",
                 "wiki_visa_parquet_root", "dump_search_incorrect_relaxed_correct",
                 "export_last_search_examples"):
        if getattr(args, name):
            setattr(args, name, str(project_path(getattr(args, name))))

    if not os.path.exists(args.inference_file):
        parser.error(f"Inference file does not exist: {args.inference_file}")

    # 评测推理结果
    stats, sample_results = evaluate_inference_results(
        args.inference_file,
        max_samples=args.max_samples,
        reference_image_root=args.reference_image_root,
        wiki_visa_parquet_root=args.wiki_visa_parquet_root,
    )

    # 打印报告
    print_evaluation_report(
        stats,
        sample_results,
        show_search_incorrect_relaxed_correct=args.show_search_incorrect_relaxed_correct,
        dump_search_incorrect_relaxed_correct=args.dump_search_incorrect_relaxed_correct,
        print_model_output=args.print_model_output,
        max_model_output_chars=args.max_model_output_chars,
        export_last_search_examples=args.export_last_search_examples,
        export_last_search_examples_all=args.export_last_search_examples_all,
    )

    # 自动保存详细结果到evaluate_results.json
    output_data = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'input_file': args.inference_file,
        'stats': _round_answer_stats_for_export(stats, ndigits=2),
        'sample_results': sample_results
    }

    # 如果指定了output_file，使用指定的文件；否则使用默认的evaluate_results.json
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = str(project_path("outputs/wiki-visa/metrics.json"))

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(record_paths(output_data, relative=True), f, indent=2, ensure_ascii=False)
        print(f"\n✅ Detailed evaluation results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving results to {output_path}: {e}")

    # 如果指定了额外的output_file，也保存一份
    if args.output_file and args.output_file != output_path:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(record_paths(output_data, relative=True), f, indent=2, ensure_ascii=False)
            print(f"📋 Additional copy saved to: {args.output_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save to additional output file: {e}")


if __name__ == "__main__":
    main()
