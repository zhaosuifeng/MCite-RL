from scripts.wiki_visa_data import INVALID_ACTION_PROMPT
import torch
import re
import numpy as np
from collections import defaultdict
import os
from project_paths import project_path
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from transformers.image_processing_base import BatchFeature
from PIL import Image
from tqdm import tqdm
import json
import ast
import math

def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(project_path(image))


    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


# Citation评测相关函数（适配generation.py的图像处理参数）
# generation.py使用的参数：MAX_PIXELS = 1536 * 28 * 28, MIN_PIXELS = 768 * 28 * 28
MAX_PIXELS_GEN = 1536 * 28 * 28  # 1,204,416
MIN_PIXELS_GEN = 768 * 28 * 28   # 602,208


def process_image_size_generation(original_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    计算图像经过process_image处理后的尺寸（适配generation.py的参数）
    
    Args:
        original_size: (width, height) 原始图像尺寸
    
    Returns:
        (width, height) 处理后的图像尺寸
    """
    original_w, original_h = original_size
    original_pixels = original_w * original_h
    
    # 如果图像太大：缩小
    if original_pixels > MAX_PIXELS_GEN:
        resize_factor = math.sqrt(MAX_PIXELS_GEN / original_pixels)
        new_width = int(original_w * resize_factor)
        new_height = int(original_h * resize_factor)
        return (new_width, new_height)
    
    # 如果图像太小：放大
    elif original_pixels < MIN_PIXELS_GEN:
        resize_factor = math.sqrt(MIN_PIXELS_GEN / original_pixels)
        new_width = int(math.ceil(original_w * resize_factor))
        new_height = int(math.ceil(original_h * resize_factor))
        # 如果取整后仍然小于MIN_PIXELS_GEN，再次调整
        new_pixels = new_width * new_height
        if new_pixels < MIN_PIXELS_GEN:
            scale_adjust = math.sqrt(MIN_PIXELS_GEN / new_pixels)
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


def build_crop_chain_from_generation(retrievaled_images: List[str], crop_history: List[Dict]) -> List[Dict]:
    """
    从generation.py的retrievaled_images和crop_history构建裁剪链
    
    Args:
        retrievaled_images: 检索到的图像路径列表
        crop_history: 裁剪历史，每个元素包含 {'bbox': [x1,y1,x2,y2], 'image_path': str, 'step': int}
    
    Returns:
        crop_chain: 裁剪链，包含每一步的源图像、bbox和结果图像
    """
    crop_chain = []
    
    # 添加检索步骤
    for idx, img_path in enumerate(retrievaled_images):
        crop_chain.append({
            'step': len(crop_chain),
            'type': 'search',
            'source_image': img_path,
            'bbox': None,
            'result_image': img_path
        })
    
    # 添加裁剪步骤
    for crop_info in crop_history:
        source_image = crop_chain[-1]['result_image'] if crop_chain else None
        crop_chain.append({
            'step': len(crop_chain),
            'type': 'crop',
            'source_image': source_image,
            'bbox': crop_info.get('bbox'),
            'result_image': crop_info.get('image_path')
        })
    
    return crop_chain


def convert_crop_coords_to_search_image_generation(citation_coords: List[int], crop_chain: List[Dict]) -> Optional[List[int]]:
    """
    将citation坐标从裁剪坐标系转换到最后一次检索图像坐标系（原始尺寸）
    适配generation.py的图像处理流程
    
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

    # 当前坐标（相对于最后一次裁剪并缩放后的图像）
    current_coords = citation_coords.copy()

    # 从最后一次裁剪开始，逐步反向转换到检索图像
    for step_idx, step in enumerate(crop_steps_after_search):
        if not step.get('bbox'):
            continue
        
        # 步骤1: 将坐标从裁剪后缩放图像（模型看到的）转换到裁剪后原始图像（保存的图像）
        # 关键修正：保存的裁剪图像是从原始图像裁剪出来的（原始尺寸），不是从缩放后的图像裁剪出来的
        cropped_image_path = step.get('result_image')
        
        # 如果裁剪图像路径不存在，尝试从上一级图像和bbox计算裁剪图像尺寸
        if not cropped_image_path or not os.path.exists(cropped_image_path):
            # 找到上一级图像（可能是检索图像或之前的裁剪图像）
            prev_image_path = None
            prev_saved_size = None
            prev_processed_size = None
            
            if step_idx < len(crop_steps_after_search) - 1:
                # 上一级是前一个裁剪步骤，向上查找直到找到存在的图像路径
                for i in range(step_idx + 1, len(crop_steps_after_search)):
                    prev_step = crop_steps_after_search[i]
                    prev_image_path = prev_step.get('result_image')
                    if prev_image_path and os.path.exists(prev_image_path):
                        prev_saved_size = get_image_size(prev_image_path)
                        if prev_saved_size:
                            prev_processed_size = process_image_size_generation(prev_saved_size)
                            break
                        prev_image_path = None
                
                # 如果还没找到，查找检索图像
                if prev_image_path is None:
                    search_image_path = last_search_step.get('source_image')
                    if search_image_path and os.path.exists(search_image_path):
                        prev_saved_size = get_image_size(search_image_path)
                        if prev_saved_size:
                            prev_processed_size = process_image_size_generation(prev_saved_size)
            else:
                # 上一级是检索图像
                search_image_path = last_search_step.get('source_image')
                if search_image_path and os.path.exists(search_image_path):
                    prev_saved_size = get_image_size(search_image_path)
                    if prev_saved_size:
                        prev_processed_size = process_image_size_generation(prev_saved_size)
            
            if prev_saved_size is None or prev_processed_size is None:
                return None
            
            # 从上一级图像和bbox计算裁剪图像尺寸
            prev_w_saved, prev_h_saved = prev_saved_size
            prev_w_proc, prev_h_proc = prev_processed_size
            prev_scale_x = prev_w_saved / prev_w_proc if prev_w_proc > 0 else 1.0
            prev_scale_y = prev_h_saved / prev_h_proc if prev_h_proc > 0 else 1.0
            
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = step['bbox']
            # 将bbox从缩放后的图像转换到原始图像
            bbox_orig_x1 = int(bbox_x1 * prev_scale_x)
            bbox_orig_y1 = int(bbox_y1 * prev_scale_y)
            bbox_orig_x2 = int(bbox_x2 * prev_scale_x)
            bbox_orig_y2 = int(bbox_y2 * prev_scale_y)
            
            # 裁剪图像的尺寸 = bbox在原始图像上的宽度和高度
            cropped_saved_size = (bbox_orig_x2 - bbox_orig_x1, bbox_orig_y2 - bbox_orig_y1)
        else:
            # 获取保存的裁剪图像尺寸（这是从原始图像裁剪出来的，原始尺寸）
            cropped_saved_size = get_image_size(cropped_image_path)
            if not cropped_saved_size:
                return None
        
        # 保存的图像经过process_image处理，得到模型看到的尺寸
        cropped_processed_size = process_image_size_generation(cropped_saved_size)
        cropped_w_saved, cropped_h_saved = cropped_saved_size
        cropped_w_proc, cropped_h_proc = cropped_processed_size
        
        # 计算缩放比例（从模型看到的缩放后图像到保存的裁剪图像）
        # 注意：保存的图像经过process_image处理时，如果像素数不在[MIN_PIXELS_GEN, MAX_PIXELS_GEN]范围内，会被缩放
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
        # bbox坐标是在上一级缩放后的图像上的坐标（模型看到的图像）
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = step['bbox']
        
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
                    prev_processed_size = process_image_size_generation(prev_saved_size)
                    prev_w_saved, prev_h_saved = prev_saved_size
                    prev_w_proc, prev_h_proc = prev_processed_size
                    
                    # 将bbox从上一级缩放后的图像转换到上一级原始图像
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
        else:
            # 上一级是检索图像
            search_image_path = last_search_step.get('source_image')
            if search_image_path and os.path.exists(search_image_path):
                search_original_size = get_image_size(search_image_path)
                if search_original_size:
                    search_processed_size = process_image_size_generation(search_original_size)
                    search_w_orig, search_h_orig = search_original_size
                    search_w_proc, search_h_proc = search_processed_size
                    
                    # 将bbox从检索图像的缩放后图像转换到检索图像的原始图像
                    search_scale_x = search_w_orig / search_w_proc if search_w_proc > 0 else 1.0
                    search_scale_y = search_h_orig / search_h_proc if search_h_proc > 0 else 1.0
                    
                    bbox_orig_x1 = int(bbox_x1 * search_scale_x)
                    bbox_orig_y1 = int(bbox_y1 * search_scale_y)
                    
                    # 当前坐标是相对于裁剪区域左上角的，需要加上bbox在原始图像上的偏移
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
                search_processed_size = process_image_size_generation(search_original_size)
                search_w_orig, search_h_orig = search_original_size
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

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int 
    num_gpus: int
    search_url: str = None
    max_model_len: int = 10240
    image_pad_id: int = 151655
    endoftext_id: int = 151643
    crop_image_output_dir: str = None  # 裁剪图像保存目录，如果为None则不保存
    checkpoint_dir: str = None  # checkpoint目录，用于保存模型输入图像


class LLMGenerationManager:
    def __init__(
        self,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id
        ))
        
        # 初始化裁剪图像输出目录
        self.crop_image_output_dir = str(project_path(config.crop_image_output_dir)) if config.crop_image_output_dir else None
        if self.crop_image_output_dir:
            os.makedirs(self.crop_image_output_dir, exist_ok=True)
        
        # 初始化模型输入图像保存目录（在checkpoint目录下）
        self.model_input_image_dir = None
        if config.checkpoint_dir:
            self.model_input_image_dir = str(project_path(config.checkpoint_dir) / 'model_input_images')
            os.makedirs(self.model_input_image_dir, exist_ok=True)

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _extract_citation(self, text: str) -> Optional[List[int]]:
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
    
    def _evaluate_citation(self, citation_coords: List[int], sample_idx: int, 
                         ground_truth_citation: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        评估citation，将模型生成的citation坐标转换到原图坐标系并计算IoU
        
        Args:
            citation_coords: 模型生成的citation坐标（相对于最后一次裁剪并缩放后的图像）
            sample_idx: 样本索引
            ground_truth_citation: 标准答案的citation坐标（相对于原始检索图像）
        
        Returns:
            评测结果字典，包含converted_citation, citation_iou等信息
        """
        result = {
            'has_citation': citation_coords is not None,
            'converted_citation': None,
            'citation_iou': None,
            'original_citation': citation_coords
        }
        
        if not citation_coords:
            return result
        
        # 构建裁剪链
        retrievaled_images = self.retrievaled_images[sample_idx] if sample_idx < len(self.retrievaled_images) else []
        crop_history = self.crop_history[sample_idx] if sample_idx < len(self.crop_history) else []
        
        # 使用与mcite.py一致的函数构建裁剪链和转换坐标
        from verl.utils.reward_score.mcite import (
            build_crop_chain_from_extra_info, 
            convert_crop_coords_to_search_image,
            calculate_iou as mcite_calculate_iou
        )
        crop_chain = build_crop_chain_from_extra_info(retrievaled_images, crop_history)
        
        # 将citation坐标转换到原始检索图像坐标系（使用与推理时一致的函数）
        converted_citation = convert_crop_coords_to_search_image(citation_coords, crop_chain)
        result['converted_citation'] = converted_citation
        
        # 添加调试信息
        if converted_citation is None:
            print(f"[Training] Warning: Failed to convert citation coordinates for sample {sample_idx}")
            print(f"  Original citation: {citation_coords}")
            print(f"  Retrievaled images: {retrievaled_images}")
            print(f"  Crop history: {crop_history}")
            print(f"  Crop chain: {crop_chain}")
        
        # 如果有ground truth citation，计算IoU（使用与推理时一致的函数）
        if ground_truth_citation and converted_citation:
            if isinstance(ground_truth_citation, str):
                try:
                    ground_truth_citation = ast.literal_eval(ground_truth_citation)
                except:
                    ground_truth_citation = None
            
            if ground_truth_citation and isinstance(ground_truth_citation, list) and len(ground_truth_citation) == 4:
                if len(converted_citation) == 4:
                    citation_iou = mcite_calculate_iou(converted_citation, ground_truth_citation)
                    result['citation_iou'] = citation_iou
                    # 添加调试信息（仅在IoU较低时打印）
                    if citation_iou < 0.5:
                        print(f"[Training] Low IoU for sample {sample_idx}: {citation_iou:.3f}")
                        print(f"  Converted citation: {converted_citation}")
                        print(f"  Ground truth citation: {ground_truth_citation}")
                else:
                    result['citation_iou'] = 0.0
            else:
                result['citation_iou'] = None
        
        return result
    
    def _postprocess_responses_first(self,batch):
        
        responses_str = self.tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        responses_str = ["<search>"+item.split('Question: ')[1].split(' \n\nassistant\n')[0]+"</search>" for item in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
        

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        def extract_tags(text):
            # 定义正则表达式，匹配 <answer>...</answer>、<search>...</search>、<think>...</think>、<bbox>...</bbox> 和 <citation>...</citation>
            pattern = r"<(answer|search|think|bbox|citation)>(.*?)</\1>"
            # 使用 findall 方法找到所有匹配的内容
            matches = re.findall(pattern, text, re.DOTALL)
            # 将匹配的内容重新组合成字符串
            result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
            return result

        responses_str = [extract_tags(resp) + self.tokenizer.eos_token for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List, rollings) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_str = []
        multi_modal_data = []
        multi_modal_inputs = []
        mm_debug = os.environ.get("MCITE_MM_DEBUG", "0").lower() in ("1", "true", "yes", "y")
        merge_length = self.processor.image_processor.merge_size**2
        # print(self.retrievaled_images)
        for idx, obs_item in enumerate(next_obs):
            # invalid
            if isinstance(obs_item,str):
                next_obs_str.append(obs_item)
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            # invalid
            elif isinstance(obs_item, list) and not isinstance(obs_item[0],dict) and len(self.retrievaled_images[idx]) == 0:
                next_obs_str.append(f'\n<|im_start|>user\n{INVALID_ACTION_PROMPT}\n<|im_end|>\n<|im_start|>assistant\n')
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            # crop
            elif isinstance(obs_item,list) and not isinstance(obs_item[0],dict):
                try:
                    latest_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][-1]
                    width, height = latest_image.size
                    raw_images_crop = Image.open(self.retrievaled_images[idx][-1])
                    raw_width, raw_height = raw_images_crop.size
                    crop_area = [int(raw_width * obs_item[0] / width), int(raw_height * obs_item[1] / height), int(raw_width * obs_item[2] / width), int(raw_height * obs_item[3] / height)]
                    crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
                    cropped_image = raw_images_crop.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))
                    input_images_list = [cropped_image]
                    raw_images_list = [process_image(image, 1536*28*28, 768*28*28) for image in input_images_list]

                    # 存储裁剪历史（用于citation坐标转换）
                    # bbox坐标是在模型看到的缩放后图像上的坐标
                    # 注意：obs_item是模型输出的坐标（在模型看到的缩放后图像上）
                    # 但我们需要存储的是在上一级图像（可能是检索图像或之前的裁剪图像）上的坐标
                    # 这里存储的是在上一级缩放后图像上的bbox坐标
                    latest_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][-1]
                    width, height = latest_image.size  # 这是模型看到的图像尺寸（缩放后）
                    crop_bbox_in_processed = obs_item  # 在模型看到的图像上的坐标
                    
                    # 保存裁剪后的图像（总是保存，用于坐标转换）
                    # 优先使用配置的目录，否则保存到checkpoint目录下
                    crop_image_path = None
                    crop_output_dir = self.crop_image_output_dir
                    if not crop_output_dir and self.model_input_image_dir:
                        # 如果没有配置crop_image_output_dir，使用checkpoint目录下的crop_images子目录
                        crop_output_dir = os.path.join(os.path.dirname(self.model_input_image_dir), 'crop_images')
                    
                    if crop_output_dir:
                        try:
                            os.makedirs(crop_output_dir, exist_ok=True)
                            # 生成唯一的文件名
                            timestamp = int(time.time() * 1000)
                            # 获取样本索引（如果有）
                            sample_indices = rollings.non_tensor_batch.get('index', None)
                            if sample_indices is not None and idx < len(sample_indices):
                                sample_idx = sample_indices[idx]
                            else:
                                sample_idx = idx
                            crop_step = len(self.crop_history[idx])
                            file_extension = '.jpg'
                            output_filename = f"sample_{sample_idx}_crop_step_{crop_step}_{timestamp}{file_extension}"
                            crop_image_path = os.path.join(crop_output_dir, output_filename)
                            # 保存原始裁剪图像（未缩放）
                            cropped_image.save(crop_image_path)
                        except Exception as e:
                            # 如果保存失败，继续使用None
                            crop_image_path = None
                            print(f"Warning: Failed to save cropped image: {e}")
                    self.crop_history[idx].append({
                        'bbox': crop_bbox_in_processed,  # 在上一级缩放后图像上的坐标
                        'image_path': crop_image_path,
                        'step': len(self.crop_history[idx])
                    })
                    
                    # 保存传入模型的图像（裁剪后的缩放图像）
                    if self.model_input_image_dir and len(raw_images_list) > 0:
                        try:
                            timestamp = int(time.time() * 1000)
                            sample_indices = rollings.non_tensor_batch.get('index', None)
                            if sample_indices is not None and idx < len(sample_indices):
                                sample_idx = sample_indices[idx]
                            else:
                                sample_idx = idx
                            crop_step = len(self.crop_history[idx]) - 1
                            input_filename = f"sample_{sample_idx}_crop_step_{crop_step}_crop_input_{timestamp}.jpg"
                            model_input_path = os.path.join(self.model_input_image_dir, input_filename)
                            raw_images_list[0].save(model_input_path)
                        except Exception as e:
                            print(f"Warning: Failed to save model input image (crop): {e}")

                    image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')
                    # Strict multimodal contract: if pixel_values missing, we must NOT emit vision tokens,
                    # otherwise later model forward can break. Downgrade to text-only observation.
                    if ('pixel_values' not in image_inputs) or ('image_grid_thw' not in image_inputs):
                        if mm_debug:
                            print(f"[MCITE_MM_DEBUG] missing keys from image_processor at idx={idx}: keys={list(getattr(image_inputs, 'data', {}).keys())}")
                        next_obs_str.append('\n<|im_start|>user\n[Warning] Image processing failed; continuing without vision input.\n<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
                        continue
                    multi_modal_data.append({'image': raw_images_list})
                    multi_modal_inputs.append(image_inputs)
                    image_grid_thw = image_inputs['image_grid_thw']
                    obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                    raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
                    obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                    next_obs_str.append(obs_str)   
                except Exception as e:
                    next_obs_str.append(f'\n<|im_start|>user\n{INVALID_ACTION_PROMPT}\n<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict())) 
            # ret image
            elif isinstance(obs_item,list) and isinstance(obs_item[0],dict):
                img_file_list = [str(project_path(item['image_file'])) for item in obs_item]
                for image_item in img_file_list:
                    if image_item not in self.retrievaled_images[idx]:
                        self.retrievaled_images[idx].append(image_item)
                        # input_images_list = img_file_list[:1]
                        input_images_list = [image_item]
                        break

                raw_images_list = [process_image(image, 1536*28*28, 768*28*28) for image in input_images_list]
                
                # 保存传入模型的图像（检索到的缩放后图像）
                if self.model_input_image_dir and len(raw_images_list) > 0:
                    try:
                        timestamp = int(time.time() * 1000)
                        sample_indices = rollings.non_tensor_batch.get('index', None)
                        if sample_indices is not None and idx < len(sample_indices):
                            sample_idx = sample_indices[idx]
                        else:
                            sample_idx = idx
                        base_name = os.path.basename(img_file_list[0]) if img_file_list else 'unknown'
                        input_filename = f"sample_{sample_idx}_search_input_{base_name}_{timestamp}.jpg"
                        model_input_path = os.path.join(self.model_input_image_dir, input_filename)
                        raw_images_list[0].save(model_input_path)
                    except Exception as e:
                        print(f"Warning: Failed to save model input image (search): {e}")

                image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')
                # Strict multimodal contract: if pixel_values missing, we must NOT emit vision tokens.
                if ('pixel_values' not in image_inputs) or ('image_grid_thw' not in image_inputs):
                    if mm_debug:
                        print(f"[MCITE_MM_DEBUG] missing keys from image_processor at idx={idx}: keys={list(getattr(image_inputs, 'data', {}).keys())}")
                    next_obs_str.append('\n<|im_start|>user\n[Warning] Retrieved images but image processing failed; continuing without vision input.\n<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
                    continue

                multi_modal_data.append({'image': raw_images_list})
                multi_modal_inputs.append(image_inputs)
                image_grid_thw = image_inputs['image_grid_thw']

                obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
                obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                next_obs_str.append(obs_str)
            else:
                raise ValueError('invalid observation')
        
        next_obs_ids = self.tokenizer(
            next_obs_str, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
    
    def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data:list, next_obs_multi_modal_inputs:list):
        mm_debug = os.environ.get("MCITE_MM_DEBUG", "0").lower() in ("1", "true", "yes", "y")

        def _has_mm_keys(x) -> bool:
            try:
                return ('pixel_values' in x) and ('image_grid_thw' in x)
            except Exception:
                return False

        if not 'multi_modal_inputs' in rollings.non_tensor_batch.keys():

            rollings.non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
            rollings.non_tensor_batch['multi_modal_data'] = np.array(next_obs_multi_modal_data, dtype=object)

            for idx, item in enumerate(next_obs_multi_modal_inputs):
                # Keep strict consistency: if we have images but missing tensors, drop images to avoid later mismatch.
                images = []
                try:
                    images = rollings.non_tensor_batch['multi_modal_data'][idx].get('image', [])
                except Exception:
                    images = []
                if images and (not _has_mm_keys(item)):
                    if mm_debug:
                        print(f"[MCITE_MM_DEBUG] init mm mismatch at idx={idx}: images={len(images)} but missing pixel_values; dropping images")
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'] = []
                    rollings.non_tensor_batch['multi_modal_inputs'][idx] = BatchFeature(dict())
                else:
                    rollings.non_tensor_batch['multi_modal_inputs'][idx] = item

        else:

            for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
                images = multi_modal_data_item.get('image', [])
                if len(images) > 0:
                    # Strict consistency: only concat when the new inputs actually contain vision tensors.
                    next_inputs = next_obs_multi_modal_inputs[idx] if idx < len(next_obs_multi_modal_inputs) else None
                    if (next_inputs is None) or (not _has_mm_keys(next_inputs)):
                        if mm_debug:
                            print(f"[MCITE_MM_DEBUG] concat mm mismatch at idx={idx}: images={len(images)} but missing pixel_values; skipping")
                        continue

                    cur_inputs = rollings.non_tensor_batch['multi_modal_inputs'][idx]
                    if _has_mm_keys(cur_inputs):
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat(
                            (cur_inputs['pixel_values'], next_inputs['pixel_values']),
                            dim=0
                        )
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat(
                            (cur_inputs['image_grid_thw'], next_inputs['image_grid_thw']),
                            dim=0
                        )
                    else:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_inputs['pixel_values']
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_inputs['image_grid_thw']

                    # data: only extend after successful tensor update to keep consistency
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(images)

        return rollings
        

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        if next_obs_ids.shape[1] != 0:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        }, rollings.non_tensor_batch)

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None and next_obs_ids.shape[1] != 0:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        padded_non_tensor_batch = {}

        padded_ids = self.tokenizer(
            ['<|im_start|>user\nHi, who are u?<|im_end|>\n<|im_start|>assistant\n'], 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        padded_ids = padded_ids[0]

        pad_input_ids = torch.full_like(active_batch.batch['input_ids'][0], self.config.endoftext_id, dtype=torch.int64) #151643
        pad_input_ids[:len(padded_ids)] = padded_ids
        pad_attention_mask = self.tensor_fn.create_attention_mask(pad_input_ids)
        pad_input_ids = pad_input_ids.unsqueeze(0)
        pad_attention_mask = pad_attention_mask.unsqueeze(0)
        pad_position_ids = self.tensor_fn.create_position_ids(pad_attention_mask)
        
        padded_batch['attention_mask'] = torch.cat([active_batch.batch['attention_mask'], pad_attention_mask.repeat(padding_size, *[1] * (len(active_batch.batch['attention_mask'].shape) - 1))], dim=0)
        padded_batch['input_ids'] = torch.cat([active_batch.batch['input_ids'], pad_input_ids.repeat(padding_size, *[1] * (len(active_batch.batch['input_ids'].shape) - 1))], dim=0)
        padded_batch['position_ids'] = torch.cat([active_batch.batch['position_ids'], pad_position_ids.repeat(padding_size, *[1] * (len(active_batch.batch['position_ids'].shape) - 1))], dim=0)
        

        for k, v in active_batch.non_tensor_batch.items():
            pad_non_tensor_item = np.empty(padding_size, dtype=object)
            if k == 'raw_prompt_ids':
                list_ids = padded_ids.tolist()
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = list_ids
            elif k == 'multi_modal_inputs':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {}
            elif k == 'multi_modal_data':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {'image': []}
            padded_non_tensor_batch[k] = np.concatenate([v, pad_non_tensor_item])
                
        padded_active_batch = DataProto.from_dict(padded_batch, padded_non_tensor_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _raw_prompt_ids(self, rollings):
        new_raw_prompt_ids = []
        rollings.batch['input_ids'] = rollings.batch['input_ids'].long()
        raw_next_obs_ids = [ids[mask == 1].tolist() for ids, mask in zip(np.array(rollings.batch['input_ids']),  np.array(rollings.batch['attention_mask']))]
        def replace_consecutive_elements(arr, target):
            result = []
            i = 0
            while i < len(arr):
                if arr[i] == target:
                    result.append(target)
                    while i + 1 < len(arr) and arr[i + 1] == target:
                        i += 1
                else:
                    result.append(arr[i])
                i += 1
            return result
        raw_next_obs_ids = [replace_consecutive_elements(row,self.config.image_pad_id) for row in raw_next_obs_ids] #151655
        raw_next_obs_ids = np.array(raw_next_obs_ids, dtype=object)
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_next_obs_ids
        return rollings

    def deactivate_batch(self, active_mask,rollings):
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        max_model_len = self.config.max_model_len
        curr_active_mask = torch.tensor([len(raw_prompt_ids_item) < max_model_len for raw_prompt_ids_item in raw_prompt_ids], dtype=torch.bool)
        active_mask = active_mask * curr_active_mask
        return active_mask

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        # original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_left_side = {'input_ids': initial_input_ids}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        # rollings_multimodal_data = gen_batch.non_tensor_batch.get('multi_modal_inputs', None)
        # rollings_multimodal_data = gen_batch.non_tensor_batch['multi_modal_inputs']
        # rollings_multimodal_data = None
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']

        self.retrievaled_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        # 存储每个样本的裁剪历史，用于构建裁剪链进行citation坐标转换
        self.crop_history = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        # 存储所有样本的citation IoU，用于计算平均值
        self.citation_ious = []

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)
            if not active_mask.sum():
                break
            
            if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                rollings_active = DataProto.from_dict(
                    tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                    non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                )
            else:
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })

            # self.processor.batch_decode(rollings_active.batch['input_ids'])
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

            # 获取所有样本的prompt文本（包括inactive的）
            all_prompt_texts = self.tokenizer.batch_decode(
                rollings.batch['input_ids'], 
                skip_special_tokens=False
            )
            
            # 获取active样本的prompt文本
            active_prompt_texts = self.tokenizer.batch_decode(
                rollings_active.batch['input_ids'], 
                skip_special_tokens=False
            )

            print(responses_str[0] if len(responses_str) > 0 else "No active responses")
            output_file = project_path("outputs/debug/output.txt")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            json_output_file = project_path("outputs/debug/output.jsonl")
            
            # 获取样本索引（如果有）
            sample_indices = rollings.non_tensor_batch.get('index', list(range(len(all_prompt_texts))))
            if isinstance(sample_indices, np.ndarray):
                sample_indices = sample_indices.tolist()
            
            # 保存为文本格式 - 只保存active样本
            with open(output_file, "a", encoding="utf-8") as f:
                active_idx_in_batch = 0
                for batch_idx in range(len(all_prompt_texts)):
                    if active_mask[batch_idx]:
                        sample_idx = sample_indices[batch_idx] if batch_idx < len(sample_indices) else batch_idx
                        f.write(f"=== Step {step} | Sample {sample_idx} | Batch {batch_idx} ===\n")
                        f.write(f"Prompt:\n{active_prompt_texts[active_idx_in_batch]}\n\n")
                        f.write(f"Response:\n{responses_str[active_idx_in_batch]}\n\n")
                        f.write("-" * 80 + "\n\n")
                        active_idx_in_batch += 1
            
            # 保存为JSON格式（每行一个JSON对象）- 只保存active样本
            # 同时将citation_converted保存到meta_info中，供reward计算使用
            if 'citation_converted' not in meta_info:
                meta_info['citation_converted'] = [None] * len(all_prompt_texts)
            if 'citation_original' not in meta_info:
                meta_info['citation_original'] = [None] * len(all_prompt_texts)
            
            with open(json_output_file, "a", encoding="utf-8") as f:
                active_idx_in_batch = 0
                for batch_idx in range(len(all_prompt_texts)):
                    if active_mask[batch_idx]:
                        sample_idx = sample_indices[batch_idx] if batch_idx < len(sample_indices) else batch_idx
                        response_text = responses_str[active_idx_in_batch]
                        # 提取citation信息 - 只在有answer时才提取（与推理评测一致）
                        # 推理时只从最终answer对应的response中提取citation，确保crop_chain完整
                        citation = None
                        if '<answer>' in response_text.lower():
                            citation = self._extract_citation(response_text)
                        data = {
                            "step": step,
                            "sample_idx": int(sample_idx),
                            "batch_idx": int(batch_idx),
                            "prompt": active_prompt_texts[active_idx_in_batch],
                            "response": response_text
                        }
                        if citation is not None:
                            data["citation"] = citation
                            # 获取ground truth citation（如果存在）
                            ground_truth_citation = None
                            if hasattr(rollings, 'non_tensor_batch') and 'ground_truth_citation' in rollings.non_tensor_batch:
                                gt_citations = rollings.non_tensor_batch['ground_truth_citation']
                                if batch_idx < len(gt_citations):
                                    ground_truth_citation = gt_citations[batch_idx]
                            # 评估citation
                            citation_eval = self._evaluate_citation(citation, batch_idx, ground_truth_citation)
                            data["citation_original"] = citation_eval['original_citation']
                            # 保存到meta_info中，供reward计算使用
                            meta_info['citation_original'][batch_idx] = citation_eval['original_citation']
                            if citation_eval['converted_citation'] is not None:
                                data["citation_converted"] = citation_eval['converted_citation']
                                meta_info['citation_converted'][batch_idx] = citation_eval['converted_citation']
                            if citation_eval['citation_iou'] is not None:
                                data["citation_iou"] = citation_eval['citation_iou']
                                # 收集IoU值用于计算平均值
                                self.citation_ious.append(citation_eval['citation_iou'])
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        active_idx_in_batch += 1

            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            # Execute in environment and process observations
            next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            next_obs_ids, next_obs_str, next_obs_multi_modal_data, next_obs_multi_modal_inputs = self._process_next_obs(next_obs, rollings)
            
            rollings = self._concat_multi_modal_data(
                rollings,
                next_obs_multi_modal_data,
                next_obs_multi_modal_inputs
            )
            
            # Update states            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )



        # final LLM rollout
        if active_mask.sum():

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)

            if active_mask.sum():

                if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })

                gen_output = self._generate_with_gpu_padding(rollings_active)

                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                
                # 获取所有样本的prompt文本（包括inactive的）
                all_prompt_texts = self.tokenizer.batch_decode(
                    rollings.batch['input_ids'], 
                    skip_special_tokens=False
                )
                
                # 获取active样本的prompt文本
                active_prompt_texts = self.tokenizer.batch_decode(
                    rollings_active.batch['input_ids'], 
                    skip_special_tokens=False
                )
                
                # 获取样本索引（如果有）
                sample_indices = rollings.non_tensor_batch.get('index', list(range(len(all_prompt_texts))))
                if isinstance(sample_indices, np.ndarray):
                    sample_indices = sample_indices.tolist()
                
                # 保存最终生成的prompt和response
                output_file = project_path("outputs/debug/output.txt")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                json_output_file = project_path("outputs/debug/output.jsonl")
                
                with open(output_file, "a", encoding="utf-8") as f:
                    active_idx_in_batch = 0
                    for batch_idx in range(len(all_prompt_texts)):
                        if active_mask[batch_idx]:
                            sample_idx = sample_indices[batch_idx] if batch_idx < len(sample_indices) else batch_idx
                            f.write(f"=== Final Step | Sample {sample_idx} | Batch {batch_idx} ===\n")
                            f.write(f"Prompt:\n{active_prompt_texts[active_idx_in_batch]}\n\n")
                            f.write(f"Response:\n{responses_str[active_idx_in_batch]}\n\n")
                            f.write("-" * 80 + "\n\n")
                            active_idx_in_batch += 1
                
                # 将citation_converted保存到meta_info中，供reward计算使用
                if 'citation_converted' not in meta_info:
                    meta_info['citation_converted'] = [None] * len(all_prompt_texts)
                if 'citation_original' not in meta_info:
                    meta_info['citation_original'] = [None] * len(all_prompt_texts)
                
                with open(json_output_file, "a", encoding="utf-8") as f:
                    active_idx_in_batch = 0
                    for batch_idx in range(len(all_prompt_texts)):
                        if active_mask[batch_idx]:
                            sample_idx = sample_indices[batch_idx] if batch_idx < len(sample_indices) else batch_idx
                            response_text = responses_str[active_idx_in_batch]
                            # 提取citation信息 - 只在有answer时才提取（与推理评测一致）
                            # 推理时只从最终answer对应的response中提取citation，确保crop_chain完整
                            citation = None
                            if '<answer>' in response_text.lower():
                                citation = self._extract_citation(response_text)
                            data = {
                                "step": "final",
                                "sample_idx": int(sample_idx),
                                "batch_idx": int(batch_idx),
                                "prompt": active_prompt_texts[active_idx_in_batch],
                                "response": response_text
                            }
                            if citation is not None:
                                data["citation"] = citation
                                # 获取ground truth citation（如果存在）
                                ground_truth_citation = None
                                if hasattr(rollings, 'non_tensor_batch') and 'ground_truth_citation' in rollings.non_tensor_batch:
                                    gt_citations = rollings.non_tensor_batch['ground_truth_citation']
                                    if batch_idx < len(gt_citations):
                                        ground_truth_citation = gt_citations[batch_idx]
                                # 评估citation
                                citation_eval = self._evaluate_citation(citation, batch_idx, ground_truth_citation)
                                data["citation_original"] = citation_eval['original_citation']
                                # 保存到meta_info中，供reward计算使用
                                meta_info['citation_original'][batch_idx] = citation_eval['original_citation']
                                if citation_eval['converted_citation'] is not None:
                                    data["citation_converted"] = citation_eval['converted_citation']
                                    meta_info['citation_converted'][batch_idx] = citation_eval['converted_citation']
                                if citation_eval['citation_iou'] is not None:
                                    data["citation_iou"] = citation_eval['citation_iou']
                                    # 收集IoU值用于计算平均值
                                    self.citation_ious.append(citation_eval['citation_iou'])
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                            active_idx_in_batch += 1
                
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                # # Execute in environment and process observations
                _, dones = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask, do_search=False
                )

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())

                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # 计算并打印平均citation IoU
        if len(self.citation_ious) > 0:
            mean_iou = sum(self.citation_ious) / len(self.citation_ious)
            print(f"Citation IoU Statistics:")
            print(f"  Total samples with citation: {len(self.citation_ious)}")
            print(f"  Mean IoU: {mean_iou:.4f}")
            if len(self.citation_ious) > 1:
                import statistics
                std_iou = statistics.stdev(self.citation_ious)
                median_iou = statistics.median(self.citation_ious)
                print(f"  Std IoU: {std_iou:.4f}")
                print(f"  Median IoU: {median_iou:.4f}")
                # IoU分布统计
                iou_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
                print(f"  IoU Distribution:")
                for min_iou, max_iou in iou_ranges:
                    count = sum(1 for iou in self.citation_ious if min_iou <= iou < max_iou)
                    percentage = count / len(self.citation_ious) * 100
                    print(f"    [{min_iou:.1f}, {max_iou:.1f}): {count} ({percentage:.1f}%)")
                high_iou_count = sum(1 for iou in self.citation_ious if iou >= 0.9)
                print(f"    [0.9, 1.0]: {high_iou_count} ({high_iou_count / len(self.citation_ious) * 100:.1f}%)")
        else:
            print("No citation IoU values to report (no valid citations found)")
        
        # =================== raw prompt ids ===================
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
        # rollings.non_tensor_batch.pop('raw_prompt_ids')
        
        if not self.is_validation:
            rollings, original_right_side = self._add_noisy_multi_modal_data(rollings, original_right_side)
        ### check again
        
        retrievaled_images_array = np.empty(len(self.retrievaled_images), dtype=object)
        for idx in range(len(self.retrievaled_images)):
            retrievaled_images_array[idx] = self.retrievaled_images[idx]
        rollings.non_tensor_batch['retrievaled_images'] = retrievaled_images_array
        
        # 保存裁剪历史，用于citation坐标转换
        crop_history_array = np.empty(len(self.crop_history), dtype=object)
        for idx in range(len(self.crop_history)):
            crop_history_array[idx] = self.crop_history[idx]
        rollings.non_tensor_batch['crop_history'] = crop_history_array
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollings)
    
    def _add_noisy_multi_modal_data(self, rollings, original_right_side):
        image_padded = Image.new('RGB', (64, 64), (0, 0, 0))

        image_padded = process_image(image_padded, 256*256, 128*128)
        image_inputs = self.processor.image_processor([image_padded], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size**2
        padded_str = f"\n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|><|im_end|>"

        padded_str_list = []
        for idx, multi_modal_item in enumerate(rollings.non_tensor_batch['multi_modal_data']):
            if len(multi_modal_item['image']) == 0:
                padded_str_list.append(padded_str)
                rollings.non_tensor_batch['multi_modal_data'][idx]['image'].append(image_padded)
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = image_inputs
            else:
                padded_str_list.append('')
            
        padded_ids = self.tokenizer(
            padded_str_list, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        original_right_side = self._update_right_side(
            original_right_side,
            padded_ids
        )
        return rollings, original_right_side


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollings) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output,rollings.non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones = [], []
        
        bbox_list = [content for action, content in zip(cur_actions, contents) if action == 'bbox']
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            if len(search_queries) > 0:
                batch_size = 100
                search_results = []
                for i in range(0, len(search_queries), batch_size):
                    batch_queries = search_queries[i:i + batch_size]
                    response = requests.get(self.config.search_url, params={"queries": batch_queries})
                    search_results_single_batch = response.json()
                    search_results.extend(search_results_single_batch)
                assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
            else:
                search_results = []
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if action == 'answer':
                    # answer会结束对话（无论是否有citation，citation是answer的一部分）
                    next_obs.append('')
                    dones.append(1)
                elif action == 'search':
                    # next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    next_obs.append(search_results.pop(0))
                    dones.append(0)
                elif action == 'bbox':
                    try:
                        bbox_value = json.loads(bbox_list.pop(0))
                        if len(bbox_value) == 4 and bbox_value[0] >= 0 and bbox_value[1] >= 0 and bbox_value[2] >= 0 and bbox_value[3] >= 0:
                            next_obs.append(bbox_value)
                        else:
                            raise ValueError("Invalid bbox value")
                    except:
                        next_obs.append(f'\n<|im_start|>user\n{INVALID_ACTION_PROMPT}\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
                else:
                    next_obs.append(f'\n<|im_start|>user\n{INVALID_ACTION_PROMPT}\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
            
        assert len(search_results) == 0

        return next_obs, dones

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # 与inference_sft_simple_new.py对齐：只检测answer/search/bbox，不单独处理citation
                # citation是answer的一部分，不需要单独识别为action
                pattern = r'<(search|answer|bbox)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

