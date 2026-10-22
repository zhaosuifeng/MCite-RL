#!/usr/bin/env python3
"""
使用vLLM API服务进行推理，保存每次生成的原始响应
不进行JSON解析，直接保存所有原始输出
支持通过base_url和api_key配置vLLM API服务
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import project_path, record_paths
if __package__:
    from .wiki_visa_data import DEFAULT_IMAGE_ROOT, iter_samples, INVALID_ACTION_PROMPT
else:
    from wiki_visa_data import DEFAULT_IMAGE_ROOT, iter_samples, INVALID_ACTION_PROMPT
import json
import argparse
import pandas as pd
import math
import ast
import re
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import requests
import tempfile
import time
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 缩放参数（与transform_data_with_resize.py一致）
MAX_PIXELS = 1536 * 28 * 28  # 401,408
MIN_PIXELS = 768 * 28 * 28  # 200,704


def process_image_size(original_size):
    """
    计算图像经过process_image处理后的尺寸
    
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
        new_pixels = new_width * new_height
        if new_pixels < MIN_PIXELS:
            scale_adjust = math.sqrt(MIN_PIXELS / new_pixels)
            new_width = int(math.ceil(new_width * scale_adjust))
            new_height = int(math.ceil(new_height * scale_adjust))
        return (new_width, new_height)
    
    # 不需要缩放
    return original_size


def get_image_size(image_path):
    """获取图像尺寸"""
    try:
        if image_path and os.path.exists(image_path):
            with Image.open(image_path) as img:
                return img.size
    except Exception:
        pass
    return None


def convert_citation_to_search_image_coords(citation_coords, all_images, last_search_image_path):
    """
    将citation坐标从最后一次裁剪图像转换到检索图像的原始尺寸坐标系
    
    Args:
        citation_coords: [x1, y1, x2, y2] 在最后一次裁剪并缩放后的图像上的坐标
        all_images: 所有图像信息列表
        last_search_image_path: 最后一次检索图像的路径（原始尺寸）
    
    Returns:
        [x1, y1, x2, y2] 在检索图像原始尺寸上的坐标，如果转换失败返回None
    """
    if not citation_coords or not all_images or not last_search_image_path:
        return None
    
    # 构建裁剪链（从all_images中提取）
    crop_steps = []
    search_steps = []
    
    for img_info in all_images:
        if img_info.get('type') == 'search_result':
            search_steps.append({
                'path': img_info.get('path'),
                'step': len(search_steps)
            })
        elif img_info.get('type') == 'crop_result':
            crop_steps.append({
                'path': img_info.get('path'),
                'bbox': img_info.get('bbox'),  # 模型生成的bbox（在缩放后的图像上）
                'step': len(crop_steps)
            })
    
    if not search_steps:
        return None
    
    # 找到最后一次检索
    last_search_step = search_steps[-1]
    
    # 获取最后一次检索之后的所有裁剪步骤（从后往前）
    crop_steps_after_search = crop_steps.copy()
    crop_steps_after_search.reverse()
    
    # 当前坐标（相对于最后一次裁剪并缩放后的图像）
    current_coords = citation_coords.copy()
    
    # 从最后一次裁剪开始，逐步反向转换到检索图像
    for step_idx, step in enumerate(crop_steps_after_search):
        if not step.get('bbox'):
            continue
        
        cropped_image_path = step.get('path')
        if not cropped_image_path or not os.path.exists(cropped_image_path):
            return None
        
        # 获取保存的裁剪图像尺寸（原始尺寸）
        cropped_saved_size = get_image_size(cropped_image_path)
        if not cropped_saved_size:
            return None
        
        # 计算缩放后的尺寸
        cropped_processed_size = process_image_size(cropped_saved_size)
        cropped_w_saved, cropped_h_saved = cropped_saved_size
        cropped_w_proc, cropped_h_proc = cropped_processed_size
        
        # 步骤1: 将坐标从缩放图像转换到原始图像
        scale_x = cropped_w_saved / cropped_w_proc if cropped_w_proc > 0 else 1.0
        scale_y = cropped_h_saved / cropped_h_proc if cropped_h_proc > 0 else 1.0
        
        current_coords = [
            int(current_coords[0] * scale_x),
            int(current_coords[1] * scale_y),
            int(current_coords[2] * scale_x),
            int(current_coords[3] * scale_y)
        ]
        
        # 步骤2: 将坐标从裁剪区域转换到上一级图像
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = step['bbox']
        
        if step_idx < len(crop_steps_after_search) - 1:
            # 上一级是前一个裁剪步骤
            prev_step = crop_steps_after_search[step_idx + 1]
            prev_image_path = prev_step.get('path')
            if prev_image_path and os.path.exists(prev_image_path):
                prev_saved_size = get_image_size(prev_image_path)
                if prev_saved_size:
                    prev_processed_size = process_image_size(prev_saved_size)
                    prev_w_saved, prev_h_saved = prev_saved_size
                    prev_w_proc, prev_h_proc = prev_processed_size
                    
                    prev_scale_x = prev_w_saved / prev_w_proc if prev_w_proc > 0 else 1.0
                    prev_scale_y = prev_h_saved / prev_h_proc if prev_h_proc > 0 else 1.0
                    
                    bbox_orig_x1 = int(bbox_x1 * prev_scale_x)
                    bbox_orig_y1 = int(bbox_y1 * prev_scale_y)
                    
                    current_coords = [
                        bbox_orig_x1 + current_coords[0],
                        bbox_orig_y1 + current_coords[1],
                        bbox_orig_x1 + current_coords[2],
                        bbox_orig_y1 + current_coords[3]
                    ]
        else:
            # 上一级是检索图像
            if last_search_image_path and os.path.exists(last_search_image_path):
                search_original_size = get_image_size(last_search_image_path)
                if search_original_size:
                    search_processed_size = process_image_size(search_original_size)
                    search_w_orig, search_h_orig = search_original_size
                    search_w_proc, search_h_proc = search_processed_size
                    
                    search_scale_x = search_w_orig / search_w_proc if search_w_proc > 0 else 1.0
                    search_scale_y = search_h_orig / search_h_proc if search_h_proc > 0 else 1.0
                    
                    bbox_orig_x1 = int(bbox_x1 * search_scale_x)
                    bbox_orig_y1 = int(bbox_y1 * search_scale_y)
                    
                    current_coords = [
                        bbox_orig_x1 + current_coords[0],
                        bbox_orig_y1 + current_coords[1],
                        bbox_orig_x1 + current_coords[2],
                        bbox_orig_y1 + current_coords[3]
                    ]
    
    # 如果没有裁剪步骤，citation坐标直接相对于检索图像的缩放后图像
    if not crop_steps_after_search:
        if last_search_image_path and os.path.exists(last_search_image_path):
            search_original_size = get_image_size(last_search_image_path)
            if search_original_size:
                search_processed_size = process_image_size(search_original_size)
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

def process_image(image):
    """
    图像缩放处理（与transform_data_with_resize.py的process_image方法一致）
    如果图像太大：缩小到max_pixels以下
    如果图像太小：放大到min_pixels以上
    保持宽高比
    """
    if isinstance(image, str):
        image = Image.open(image)

    original_w, original_h = image.size
    original_pixels = original_w * original_h

    # 如果图像太大：缩小
    if original_pixels > MAX_PIXELS:
        resize_factor = math.sqrt(MAX_PIXELS / original_pixels)
        new_width = int(original_w * resize_factor)
        new_height = int(original_h * resize_factor)
        image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    # 如果图像太小：放大
    elif original_pixels < MIN_PIXELS:
        resize_factor = math.sqrt(MIN_PIXELS / original_pixels)
        # 使用向上取整确保达到MIN_PIXELS
        new_width = int(math.ceil(original_w * resize_factor))
        new_height = int(math.ceil(original_h * resize_factor))
        # 如果取整后仍然小于MIN_PIXELS，再次调整
        new_pixels = new_width * new_height
        if new_pixels < MIN_PIXELS:
            # 再次应用缩放因子
            scale_adjust = math.sqrt(MIN_PIXELS / new_pixels)
            new_width = int(math.ceil(new_width * scale_adjust))
            new_height = int(math.ceil(new_height * scale_adjust))
        image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    # 转换为RGB模式（如果需要）
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image
    
# 配置
SEARCH_ENGINE_URL = 'http://localhost:8002/search'
RAW_IMAGE_DIR = str(project_path(DEFAULT_IMAGE_ROOT))
IMAGE_OUTPUT_DIR = str(project_path("outputs/wiki-visa/image_crop"))
# 保存传入模型的图像（已缩放后的图像）
MODEL_INPUT_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_OUTPUT_DIR), 'model_inputs')

# Prompt（与训练数据格式完全一致，使用XML格式而非JSON）
# 注意：训练数据使用 <think> 和 <search> 标签，不使用JSON格式
#USER_PROMPT_TEMPLATE = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. If you think that a specific region of the last image could help answer the question, please use crop tool to provide a detailed view of the relevant region, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the output, including the answer and the final bounding box as visual citation. The answer should be provided inside <answer> and </answer>, and visual citation should be provided inside <citation> and </citation>. IMPORTANT: The citation coordinates must be relative to the LAST CROPPED IMAGE you see (not the original search image). If you have cropped the image, the citation should be based on the coordinates of the most recent cropped image. For example, <answer> Beijing </answer><citation>[100, 100, 200, 200]</citation>. Question: {question}"""


def resolve_image_path(image_path):
    """Resolve project paths first, then look up page IDs under image_root."""
    if not image_path:
        return None
    path = project_path(image_path)
    if path.is_file():
        return str(path)
    image_root = project_path(RAW_IMAGE_DIR)
    candidate = image_root / Path(image_path).name
    if candidate.is_file():
        return str(candidate)
    image_id = Path(image_path).stem
    for candidate in image_root.rglob("*"):
        if (candidate.is_file() and candidate.stem == image_id
                and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}):
            return str(candidate)
    return None


def crop_and_dump(image, bbox, output_folder=IMAGE_OUTPUT_DIR, uid_prefix=None):
    """裁剪图片并保存"""
    if not output_folder:
        return None
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if isinstance(image, Image.Image):
        img = image
        file_extension = '.jpg'
    else:
        resolved_path = resolve_image_path(image)
        if resolved_path is None:
            return None
        try:
            img = Image.open(resolved_path)
        except Exception as e:
            return None
        file_extension = os.path.splitext(resolved_path)[1] or '.jpg'
    
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # 确保坐标在图像范围内
    width, height = img.size
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(x1, min(x2, width))
    y2 = max(y1, min(y2, height))
    
    # 裁剪图像（保存原始裁剪图像，不进行缩放处理）
    # 缩放处理会在发送给模型时进行
    cropped_image = img.crop((x1, y1, x2, y2))
    
    timestamp = int(time.time() * 1000)
    image_id = f"img_{timestamp}"
    
    if uid_prefix:
        output_filename = f"{uid_prefix}_{image_id}_{timestamp}{file_extension}"
    else:
        output_filename = f"{image_id}_{timestamp}{file_extension}"
    output_path = os.path.join(output_folder, output_filename)
    
    try:
        cropped_image.save(output_path)
        return output_path
    except Exception as e:
        return None


def load_parquet_data(parquet_path: str, max_samples=None):
    """Load original HF or prepared Wiki-VISA rows without image bytes."""
    return list(iter_samples([parquet_path], max_samples=max_samples))


class SFTModelInference:
    """使用vLLM API服务的模型推理类"""
    
    def __init__(self, base_url: str, api_key: str, model_name: str = None):
        """
        初始化OpenAI客户端（连接vLLM服务）
        
        Args:
            base_url: vLLM API服务的基础URL（例如：http://localhost:8000/v1）
            api_key: API密钥（vLLM服务可能不需要，但可以设置为任意值）
            model_name: 模型名称，如果为None则使用服务默认模型
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name or "default"  # vLLM服务可能使用默认模型名
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def _encode_image_to_base64(self, image):
        """
        将图像编码为base64字符串
        
        Args:
            image: PIL Image对象或图像路径
            
        Returns:
            base64编码的字符串（包含data URI前缀）
        """
        if isinstance(image, str):
            # 如果是路径，打开图像
            image = Image.open(image)
        
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 将图像编码为base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
    
    def _convert_messages_to_openai_format(self, messages: List[Dict]):
        """
        将消息格式转换为OpenAI API格式（vLLM兼容）
        
        Args:
            messages: 原始消息列表，格式为 [{"role": "user/assistant", "content": ...}]
            
        Returns:
            OpenAI API格式的消息列表
        """
        openai_messages = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # 如果content是列表（包含图像和文本）
            if isinstance(content, list):
                openai_content = []
                for item in content:
                    if isinstance(item, dict):
                        if 'image' in item or ('type' in item and item.get('type') == 'image'):
                            # 处理图像
                            img = item.get('image') or item.get('image_path')
                            if img:
                                try:
                                    img_base64 = self._encode_image_to_base64(img)
                                    openai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": img_base64
                                        }
                                    })
                                except Exception as e:
                                    print(f"⚠️ Failed to encode image: {e}")
                        elif 'text' in item:
                            # 处理文本
                            openai_content.append({
                                "type": "text",
                                "text": item['text']
                            })
                        elif 'type' in item and item['type'] == 'text':
                            openai_content.append({
                                "type": "text",
                                "text": item.get('text', '')
                            })
                openai_messages.append({
                    "role": role,
                    "content": openai_content
                })
            else:
                # 纯文本内容
                openai_messages.append({
                    "role": role,
                    "content": str(content)
                })
        
        return openai_messages

    # def _log_visual_tokens(self, inputs):
    #     """
    #     统计并打印当前batch中，每个样本对应的视觉token数量。
    #     计算方式与训练侧generation.py中基于 image_grid_thw 和 merge_size 的逻辑保持一致。
    #     """
    #     try:
    #         if "image_grid_thw" not in inputs:
    #             return
    #         image_grid_thw = inputs["image_grid_thw"]

    #         # 部分processor会挂载image_processor并提供merge_size（与训练侧一致）
    #         image_processor = getattr(self.processor, "image_processor", None)
    #         merge_size = getattr(image_processor, "merge_size", None) if image_processor is not None else None

    #         if merge_size is None:
    #             # 没有merge_size时，至少打印原始grid信息
    #             print(f"[VISUAL TOKENS] image_grid_thw={image_grid_thw.tolist()} (merge_size not found, raw grid only)")
    #             return

    #         merge_length = merge_size ** 2
    #         visual_tokens_per_sample = []

    #         # 调试：打印实际的数据形状和内容
    #         print(f"[VISUAL TOKENS DEBUG] image_grid_thw.shape={image_grid_thw.shape}, "
    #               f"image_grid_thw={image_grid_thw.tolist()}, "
    #               f"merge_size={merge_size}, merge_length={merge_length}")

    #         # 处理不同的形状情况
    #         if image_grid_thw.dim() == 2:
    #             # 形状为 (num_images, 3)，与训练时一致
    #             total = 0
    #             for grid in image_grid_thw:
    #                 grid_prod = grid.prod().item()
    #                 tokens = int(grid_prod // merge_length)
    #                 total += tokens
    #                 print(f"[VISUAL TOKENS DEBUG] grid={grid.tolist()}, grid_prod={grid_prod}, tokens={tokens}")
    #             visual_tokens_per_sample.append(total)
    #         elif image_grid_thw.dim() == 3:
    #             # 形状为 (batch, num_images, 3)
    #             for b in range(image_grid_thw.shape[0]):
    #                 total = 0
    #                 for grid in image_grid_thw[b]:
    #                     grid_prod = grid.prod().item()
    #                     tokens = int(grid_prod // merge_length)
    #                     total += tokens
    #                     print(f"[VISUAL TOKENS DEBUG] batch={b}, grid={grid.tolist()}, grid_prod={grid_prod}, tokens={tokens}")
    #                 visual_tokens_per_sample.append(total)
    #         else:
    #             # 其他形状，尝试直接计算
    #             total = 0
    #             if image_grid_thw.numel() == 3:
    #                 # 单个grid [T, H, W]
    #                 grid_prod = image_grid_thw.prod().item()
    #                 total = int(grid_prod // merge_length)
    #                 print(f"[VISUAL TOKENS DEBUG] single grid={image_grid_thw.tolist()}, grid_prod={grid_prod}, tokens={total}")
    #             visual_tokens_per_sample.append(total)

    #         print(f"[VISUAL TOKENS] image_grid_thw={image_grid_thw.tolist()}, "
    #               f"merge_length={merge_length}, "
    #               f"visual_tokens_per_sample={visual_tokens_per_sample}")
    #     except Exception as e:
    #         print(f"[VISUAL TOKENS] Failed to log visual tokens: {e}")

    def generate(self, messages: List[Dict], max_new_tokens: int = 2048, temperature: float = 0.7):
        """
        生成响应（使用vLLM API服务）
        
        Args:
            messages: 消息列表
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            生成的响应文本
        """
        try:
            # 转换消息格式为OpenAI API格式
            openai_messages = self._convert_messages_to_openai_format(messages)
            
            # 调用vLLM API服务
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # 提取响应文本
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return ""
                
        except Exception as e:
            print(f"❌ Error calling vLLM API: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_with_search(self, sample: Dict, max_iterations: int = 10, **kwargs):
        """支持多轮对话的生成（search -> crop -> answer），保存所有原始响应"""
        query = sample['query']
        uid = sample.get('uid', 'unknown')
        
        messages = []
        all_responses = []  # 保存所有原始响应
        all_images = []  # 保存图像信息
        previous_search_image_paths = []  # 记录之前检索过的图像路径，避免重复
        
        # 保存当前用于裁剪的缩放后图像（模型看到的图像）
        # 第一次裁剪：缩放后的检索图像
        # 后续裁剪：上一次裁剪后再次缩放的图像
        current_processed_image = None  # PIL Image对象，经过process_image处理的图像
        current_raw_image_path = None   # 原始图像路径（用于坐标映射 - 与训练逻辑一致）
        
        # 使用与训练数据完全一致的格式（不使用system prompt，直接用user prompt）
        # 训练数据格式：完整的指令 + Question: {question}
        user_content = query
        print(user_content)
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            try:
                # 生成响应
                response = self.generate(messages=messages, **kwargs)
                
                # 保存原始响应
                all_responses.append({
                    'iteration': iteration,
                    'response': response,
                    'messages_before': len(messages)
                })
                
                print(f"[{uid}] Iteration {iteration}/{max_iterations} - Response length: {len(response)} chars")
                print(f"[{uid}] Response preview (first 200 chars): {response[:200]}")
                
                # 简单的关键词检测，决定下一步动作（不强制解析JSON）
                should_continue = True
                has_search = False
                has_crop = False
                has_answer = False
                
                # 检查是否有answer关键词（优先检查，如果已经有答案就直接退出）
                if '<answer>' in response.lower() or '"answer"' in response.lower():
                    has_answer = True
                    print(f"[{uid}] ✅ Detected answer")
                    # 保存响应后停止
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    break
                
                # 检查是否有search关键词
                if '<search>' in response.lower() or '"search"' in response.lower():
                    has_search = True
                    print(f"[{uid}] 🔍 Detected search action")
                    print(f"[{uid}] 📝 Raw response for search extraction (first 500 chars): {response[:500]}")
                    
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # 调用检索API
                    try:
                        # 从response中提取search query（简化版）
                        search_query = None
                        
                        # 方式1: 尝试XML格式 <search>...</search>
                        search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
                        if search_match:
                            search_query = search_match.group(1).strip()
                            print(f"[{uid}] 🔍 Extracted search query from XML: {search_query[:100]}...")
                        
                        # 方式2: 尝试JSON格式
                        if not search_query and '"search"' in response:
                            try:
                                # 尝试提取JSON
                                start = response.find('{')
                                end = response.rfind('}') + 1
                                if start >= 0 and end > start:
                                    json_str = response[start:end]
                                    parsed = json.loads(json_str)
                                    search_query = parsed.get('search', '')
                                    if search_query:
                                        search_query = search_query.strip()
                                        print(f"[{uid}] 🔍 Extracted search query from JSON: {search_query[:100]}...")
                            except Exception as e:
                                print(f"[{uid}] ⚠️ Failed to parse JSON for search query: {e}")
                        
                        # 方式3: 如果提取失败，使用原始query
                        if not search_query or search_query.lower() == 'query':
                            print(f"[{uid}] ⚠️ Search query extraction failed or got 'query', using original query")
                            search_query = query
                            print(f"[{uid}] 🔍 Using original query as search query: {search_query[:100]}...")
                        
                        # 确保search_query不是空字符串或只是"query"这个词
                        if not search_query or search_query.strip().lower() == 'query':
                            print(f"[{uid}] ⚠️ Warning: Search query is empty or just 'query', using original query")
                            search_query = query
                        
                        # 最终确认并打印
                        print(f"[{uid}] 🔍 Final search query to use: {search_query}")
                        print(f"[{uid}] 🔍 Search query length: {len(search_query)}")
                        if len(search_query) < 10:
                            print(f"[{uid}] ⚠️ Warning: Search query is very short, might be wrong!")
                        
                        search_result = requests.get(
                            SEARCH_ENGINE_URL, 
                            params={"queries": [search_query]},
                            timeout=30
                        )
                        
                        search_result.raise_for_status()
                        search_result_json = search_result.json()
                        print(f"[{uid}] 🔍 Raw search_result_json: {search_result_json}")
                        # 处理搜索结果（只使用检索排序的 top-1，不读取参考答案或参考页）
                        image_path_list = []
                        # 当前检索服务返回格式类似：
                        # {"request_id": None, "results": [[{...}, {...}, ...]]}
                        results_list = search_result_json.get("results", []) if isinstance(search_result_json, dict) else search_result_json
                        if results_list and isinstance(results_list[0], list):
                            base_list = results_list[0]
                        else:
                            base_list = results_list or []
                        all_candidates = base_list[:1] if base_list else []  # 只取前1个候选
                        
                        # 初始化变量到循环外
                        selected_image_path = None
                        selected_image_id = None
                        selected_item = None
                        
                        # 辅助函数：检查图像是否已使用
                        def is_used(candidate_path):
                            """检查图像是否已使用过"""
                            if not candidate_path:
                                return False
                            resolved = resolve_image_path(candidate_path)
                            return resolved and resolved in previous_search_image_paths
                        
                        # 选择排序最高且未使用的图像
                        if selected_item is None:
                            for idx, item in enumerate(all_candidates):
                                candidate_path = None
                                if 'image_path' in item:
                                    candidate_path = item['image_path']
                                elif 'image_file' in item:
                                    candidate_path = item['image_file']
                                
                                if candidate_path and not is_used(candidate_path):
                                    resolved_candidate = resolve_image_path(candidate_path)
                                    if resolved_candidate:
                                        selected_item = item
                                        selected_image_path = candidate_path
                                        previous_search_image_paths.append(resolved_candidate)
                                        print(f"[{uid}] 🔍 Selected image (rank {idx + 1}, highest rank not used): {os.path.basename(resolved_candidate)}")
                                        break
                        
                        # 策略3: 如果候选使用过，则使用top1（至少保证有结果）
                        if selected_item is None and all_candidates:
                            selected_item = all_candidates[0]
                            if 'image_path' in selected_item:
                                selected_image_path = selected_item['image_path']
                            elif 'image_file' in selected_item:
                                selected_image_path = selected_item['image_file']
                            resolved_path = resolve_image_path(selected_image_path)
                            if resolved_path:
                                previous_search_image_paths.append(resolved_path)
                            print(f"[{uid}] ⚠️ Top-1 candidate was used before, using it anyway: {os.path.basename(resolved_path) if resolved_path else 'N/A'}")
                        
                        # 处理选中的图像
                        items_to_process = [selected_item] if selected_item else []
                        
                        for item in items_to_process:
                            if 'image_path' in item:
                                selected_image_path = item['image_path']
                                image_path_list.append(selected_image_path)
                                # 优先从item获取image_id
                                if 'image_id' in item:
                                    selected_image_id = str(item['image_id'])  # 确保是字符串
                                    print(f"[{uid}] 🔍 Found image_id in API response: {selected_image_id}")
                            elif 'image_file' in item:
                                image_data = item['image_file']
                                if image_data.startswith('data:image/jpeg;base64,'):
                                    import base64
                                    image_data = image_data.split(',')[1]
                                    image_bytes = base64.b64decode(image_data)
                                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                                    temp_file.write(image_bytes)
                                    temp_file.close()
                                    selected_image_path = temp_file.name
                                    image_path_list.append(selected_image_path)
                                else:
                                    selected_image_path = image_data
                                    image_path_list.append(selected_image_path)
                            
                            # 如果还没有image_id，再次尝试从item获取
                            if selected_image_id is None and 'image_id' in item:
                                selected_image_id = str(item['image_id'])
                                print(f"[{uid}] 🔍 Found image_id (second attempt): {selected_image_id}")
                        
                        if image_path_list and selected_image_path:
                            resolved_path = resolve_image_path(selected_image_path)
                            
                            if resolved_path and os.path.exists(resolved_path):
                                selected_image = Image.open(resolved_path).convert("RGB")
                                
                                # 应用图像缩放处理（与generation.py一致）
                                selected_image_processed = process_image(selected_image)
                                
                                # 保存当前用于裁剪的缩放后图像（第一次裁剪时使用）
                                current_processed_image = selected_image_processed.copy()
                                # 同时保存原始图像路径（用于坐标映射 - 与训练逻辑一致）
                                current_raw_image_path = resolved_path
                                
                                print(f"[{uid}] 🔍 Image processed: original={selected_image.size}, processed={selected_image_processed.size}")
                                
                                # 保存传入模型的图像（检索到的缩放后图像）
                                try:
                                    os.makedirs(MODEL_INPUT_IMAGE_DIR, exist_ok=True)
                                    ts = int(time.time() * 1000)
                                    base_name = os.path.basename(resolved_path)
                                    input_filename = f"{uid}_iter{iteration}_search_input_{base_name}_{ts}.jpg"
                                    model_input_path = os.path.join(MODEL_INPUT_IMAGE_DIR, input_filename)
                                    selected_image_processed.save(model_input_path)
                                    
                                    # 记录到 all_images，便于后续分析
                                    all_images.append({
                                        'iteration': iteration,
                                        'type': 'model_input_search',
                                        'path': model_input_path,
                                        'source_path': resolved_path,
                                        'image_id': selected_image_id
                                    })
                                    print(f"[{uid}] 💾 Saved model input image (search): {model_input_path}")
                                except Exception as e:
                                    print(f"[{uid}] ⚠️ Failed to save model input image (search): {e}")
                                
                                # 如果还没有image_id，从路径提取
                                if selected_image_id is None:
                                    filename = os.path.basename(resolved_path)
                                    name_without_ext = os.path.splitext(filename)[0]
                                    
                                    # 尝试多种方式提取ID
                                    # 方式1: 纯数字文件名
                                    if name_without_ext.isdigit():
                                        selected_image_id = name_without_ext
                                    # 方式2: 包含下划线，提取数字部分
                                    elif '_' in name_without_ext:
                                        parts = name_without_ext.split('_')
                                        # 尝试最后一个部分
                                        if parts[-1].isdigit():
                                            selected_image_id = parts[-1]
                                        # 或者提取所有数字，取最长的
                                        else:
                                            numbers = re.findall(r'\d+', name_without_ext)
                                            if numbers:
                                                selected_image_id = max(numbers, key=len)
                                    # 方式3: 提取所有数字
                                    else:
                                        numbers = re.findall(r'\d+', name_without_ext)
                                        if numbers:
                                            selected_image_id = max(numbers, key=len)
                                    
                                    if selected_image_id:
                                        print(f"[{uid}] 🔍 Extracted image_id from path: {selected_image_id} (from {filename})")
                                
                                print(f"[{uid}] 🔍 Final image_id for retrieval: {selected_image_id}")
                                
                                all_images.append({
                                    'iteration': iteration,
                                    'type': 'search_result',
                                    'path': resolved_path,
                                    'image_id': selected_image_id
                                })
                                
                                messages.append({
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": selected_image_processed}
                                    ]
                                })
                                print(messages)
                            else:
                                print(f"[{uid}] ⚠️ Could not resolve image path: {selected_image_path}")
                                break
                        else:
                            print(f"[{uid}] ⚠️ No images in search result")
                            break
                    except Exception as e:
                        print(f"[{uid}] ❌ Search API error: {e}")
                        break
                
                # 检查是否有crop关键词
                elif '<bbox>' in response.lower() or '"bbox"' in response.lower():
                    has_crop = True
                    print(f"[{uid}] 📦 Detected crop action")

                    # 提取bbox坐标
                    bbox_match = re.search(r'<bbox>\s*\[([^\]]+)\]\s*</bbox>', response)
                    bbox_coords = None
                    if bbox_match:
                        try:
                            bbox_coords = ast.literal_eval(f"[{bbox_match.group(1)}]")
                        except:
                            pass

                    if not bbox_coords:
                        print(f"[{uid}] ⚠️ Could not extract bbox coordinates from response")
                        break

                    if not all_images:
                        print(f"[{uid}] ⚠️ No image to crop")
                        break

                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    print(messages)
                    # 实际执行裁剪操作
                    try:
                        # ============ 关键修复：与训练逻辑一致 ============
                        # 训练时的逻辑：将模型生成的bbox（基于缩放后的图像）映射回原始图像，
                        # 然后在原始图像上裁剪，最后再缩放
                        
                        if current_processed_image is None or current_raw_image_path is None:
                            print(f"[{uid}] ⚠️ No processed image or raw image path available for cropping")
                            break
                        
                        # 1. 获取缩放后的图像尺寸（模型看到的）
                        width, height = current_processed_image.size
                        
                        # 2. 获取原始图像尺寸
                        raw_image = Image.open(current_raw_image_path)
                        raw_width, raw_height = raw_image.size
                        
                        # 3. 提取bbox坐标
                        x1, y1, x2, y2 = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                        
                        print(f"[{uid}] 📦 Cropping with coordinate mapping:")
                        print(f"  Processed image size (model sees): {width}x{height}")
                        print(f"  Raw image size: {raw_width}x{raw_height}")
                        print(f"  Model generated bbox: [{x1}, {y1}, {x2}, {y2}]")
                        
                        # 4. 将bbox坐标从缩放后的图像映射回原始图像（与训练一致）
                        crop_area = [
                            int(raw_width * x1 / width),
                            int(raw_height * y1 / height),
                            int(raw_width * x2 / width),
                            int(raw_height * y2 / height)
                        ]
                        
                        # 5. 边界检查（在原始图像尺寸范围内）
                        crop_area = [
                            max(0, crop_area[0]),
                            max(0, crop_area[1]),
                            min(raw_width, crop_area[2]),
                            min(raw_height, crop_area[3])
                        ]
                        
                        print(f"  Mapped to raw image: [{crop_area[0]}, {crop_area[1]}, {crop_area[2]}, {crop_area[3]}]")
                        
                        # 6. 在原始图像上裁剪（与训练一致）
                        cropped_image = raw_image.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))
                        
                        print(f"[{uid}] 📦 Cropped from raw image, size: {cropped_image.size}")
                        
                        # 保存裁剪后的原始图像（未缩放）
                        timestamp = int(time.time() * 1000)
                        image_id = f"img_{timestamp}"
                        file_extension = '.jpg'
                        
                        if uid:
                            output_filename = f"{uid}_{image_id}_{timestamp}{file_extension}"
                        else:
                            output_filename = f"{image_id}_{timestamp}{file_extension}"
                        cropped_image_path = os.path.join(IMAGE_OUTPUT_DIR, output_filename)
                        
                        # 确保输出目录存在
                        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
                        
                        try:
                            cropped_image.save(cropped_image_path)
                            print(f"[{uid}] 📦 Cropped image saved: {cropped_image_path}")
                            
                            # 7. 对裁剪后的图像应用缩放处理（与训练一致）
                            cropped_processed = process_image(cropped_image.copy())
                            
                            print(f"[{uid}] 📦 Cropped image processed: original={cropped_image.size}, processed={cropped_processed.size}")
                            
                            # 8. 更新当前图像状态（下一次裁剪时使用）
                            current_processed_image = cropped_processed.copy()
                            # 保存裁剪后的原图路径，用于后续裁剪的坐标映射
                            current_raw_image_path = cropped_image_path

                            # 将裁剪结果添加到images列表
                            all_images.append({
                                'iteration': iteration,
                                'type': 'crop_result',
                                'path': cropped_image_path,
                            'bbox': bbox_coords  # 保存模型生成的原始坐标（用于调试）
                            })

                            # 保存传入模型的图像（裁剪后的缩放图像）
                            try:
                                os.makedirs(MODEL_INPUT_IMAGE_DIR, exist_ok=True)
                                ts = int(time.time() * 1000)
                                input_filename = f"{uid}_iter{iteration}_crop_input_{ts}.jpg"
                                model_input_path = os.path.join(MODEL_INPUT_IMAGE_DIR, input_filename)
                                cropped_processed.save(model_input_path)
                                
                                all_images.append({
                                    'iteration': iteration,
                                    'type': 'model_input_crop',
                                    'path': model_input_path,
                                    'source_crop_path': cropped_image_path,
                                    'bbox': bbox_coords
                                })
                                print(f"[{uid}] 💾 Saved model input image (crop): {model_input_path}")
                            except Exception as e:
                                print(f"[{uid}] ⚠️ Failed to save model input image (crop): {e}")

                            # 使用裁剪后的图像继续对话（使用处理后的图像，与模型看到的图像一致）
                            messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": cropped_processed}
                                ]
                            })
                        except Exception as e:
                            print(f"[{uid}] ❌ Failed to save cropped image: {e}")
                            print(f"[{uid}] ❌ Failed to crop image")
                            # fallback到文本提示
                            messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Examine this cropped image. Continue cropping if needed, or provide your final answer if you can see the answer clearly. Remember: if you provide a citation, it must be relative to the last cropped image you see."}
                                ]
                            })
                    except Exception as e:
                        print(f"[{uid}] ❌ Error during cropping: {e}")
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Examine this cropped image. Continue cropping if needed, or provide your final answer if you can see the answer clearly. Remember: if you provide a citation, it must be relative to the last cropped image you see."}
                            ]
                        })
                
                else:
                    # 无法识别动作，记录并继续或停止
                    print(f"[{uid}] ⚠️ Unknown response format, saving and continuing...")
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    # 可以选择继续或停止
                    # break  # 如果选择停止
                    # 或者继续，添加一个提示
                    messages.append({
                        "role": "user",
                        "content": INVALID_ACTION_PROMPT
                    })
                    
            except Exception as e:
                print(f"[{uid}] ❌ Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                all_responses.append({
                    'iteration': iteration,
                    'response': '',
                    'error': str(e)
                })
                break
        
        # 提取citation并转换坐标
        final_citation = None
        final_citation_converted = None  # 转换到检索图像原始尺寸的坐标
        last_search_image_path = None
        
        # 找到最后一次检索的图像路径
        for img_info in all_images:
            if img_info.get('type') == 'search_result':
                path = img_info.get('path')
                if path:
                    last_search_image_path = path
        
        # 从所有响应中提取citation（从后往前）
        for response_info in reversed(all_responses):
            response = response_info.get('response', '')
            if not response:
                continue
            
            # 方法1: 标准格式 <citation>[x1, y1, x2, y2]</citation>
            citation_match = re.search(r'<citation>\s*\[([^\]]+)\]\s*</citation>', response, re.IGNORECASE | re.DOTALL)
            if citation_match:
                try:
                    final_citation = ast.literal_eval(f"[{citation_match.group(1)}]")
                    if isinstance(final_citation, list) and len(final_citation) == 4:
                        break
                except:
                    pass
            
            # 方法2: 查找 </answer> 后面的坐标
            answer_end_match = re.search(r'</answer>', response, re.IGNORECASE)
            if answer_end_match:
                after_answer = response[answer_end_match.end():]
                coord_match = re.search(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', after_answer)
                if coord_match:
                    try:
                        final_citation = [int(coord_match.group(1)), int(coord_match.group(2)), 
                                         int(coord_match.group(3)), int(coord_match.group(4))]
                        break
                    except:
                        pass
            
            # 方法3: 在response末尾查找坐标
            all_coords = re.findall(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', response)
            if all_coords:
                try:
                    last_coord = all_coords[-1]
                    final_citation = [int(last_coord[0]), int(last_coord[1]), 
                                     int(last_coord[2]), int(last_coord[3])]
                    break
                except:
                    pass
        
        # 如果找到了citation，进行坐标转换
        if final_citation and last_search_image_path:
            try:
                final_citation_converted = convert_citation_to_search_image_coords(
                    final_citation, all_images, last_search_image_path
                )
                if final_citation_converted:
                    print(f"[{uid}] 📍 Citation converted: {final_citation} -> {final_citation_converted}")
                else:
                    print(f"[{uid}] ⚠️ Failed to convert citation coordinates")
            except Exception as e:
                print(f"[{uid}] ⚠️ Error converting citation coordinates: {e}")
                import traceback
                traceback.print_exc()
        
        # 整理返回结果
        result = {
            'uid': uid,
            'query': query,
            'all_responses': all_responses,  # 保存所有原始响应
            'all_images': all_images,
            'iterations': iteration,
            'completed': any(re.search(r'<answer>\s*\S.*?</answer>|"answer"\s*:\s*"[^"\s][^"]*"',
                                       r.get('response', ''), re.IGNORECASE | re.DOTALL)
                             is not None for r in all_responses),
            # 保存citation信息
            'citation_original': final_citation,  # 模型生成的原始坐标（相对于最后一次裁剪图像）
            'citation_converted': final_citation_converted  # 转换后的坐标（相对于检索图像原始尺寸）
        }
        
        # 保存reference信息（用于后续评估）
        if 'reference_answer' in sample:
            result['reference_answer'] = sample['reference_answer']
        if 'bounding_box' in sample:
            result['ground_truth_citation'] = sample['bounding_box']
        if 'reference_page' in sample:
            result['reference_page'] = sample['reference_page']
        
        return record_paths(result, relative=True)
    
    def batch_generate(self, samples: List[Dict], output_file: str = None, workers_num: int = 1, **kwargs):
        """批量生成，支持断点续写和多线程
        
        Args:
            samples: 样本列表
            output_file: 输出文件路径
            workers_num: 线程数（1表示单线程，>1表示多线程）
            **kwargs: 传递给generate_with_search的其他参数
        """
        processed_uids = set()
        
        # 检查已处理的样本
        if output_file and os.path.exists(output_file):
            print(f"Found existing output file: {output_file}")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            result = json.loads(line)
                            if 'uid' in result:
                                processed_uids.add(result['uid'])
                        except json.JSONDecodeError:
                            pass
                print(f"Already processed: {len(processed_uids)} samples")
            except Exception as e:
                print(f"Warning: Could not read existing file: {e}")
        
        # 过滤已处理的样本
        samples_to_process = [s for s in samples if s.get('uid') not in processed_uids]
        skipped_count = len(samples) - len(samples_to_process)
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed samples")
            print(f"Remaining samples to process: {len(samples_to_process)}")
        
        if len(samples_to_process) == 0:
            print("⚠️ No samples to process. All samples have already been processed.")
            return
        
        # 创建输出目录
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # 文件写入锁（用于多线程）
        file_lock = threading.Lock()
        
        def process_single_sample(sample):
            """处理单个样本的包装函数"""
            try:
                result = self.generate_with_search(sample, **kwargs)
                
                print(f"\n[Sample {sample.get('uid', 'unknown')}]")
                print(f"Query: {sample['query'][:100]}..." if len(sample['query']) > 100 else f"Query: {sample['query']}")
                print(f"Total iterations: {result.get('iterations', 0)}")
                print(f"Total responses saved: {len(result.get('all_responses', []))}")
                
                if result.get('completed'):
                    print(f"✅ Completed with answer detected")
                else:
                    print(f"⚠️ Stopped without clear answer")
                
                return result
                
            except Exception as e:
                print(f"\n[Sample {sample.get('uid', 'unknown')}] Error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'uid': sample.get('uid', 'unknown'),
                    'query': sample.get('query', ''),
                    'all_responses': [],
                    'error': str(e)
                }
        
        def write_result(result, output_file, file_lock):
            """线程安全的结果写入函数"""
            if result and output_file:
                try:
                    with file_lock:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            f.flush()  # 立即刷新到磁盘
                except Exception as e:
                    print(f"Error writing result for {result.get('uid', 'unknown')}: {e}")
        
        # 单线程处理
        if workers_num == 1:
            for sample in tqdm(samples_to_process, desc="Generating"):
                result = process_single_sample(sample)
                write_result(result, output_file, file_lock)
        
        # 多线程处理
        else:
            print(f"Using {workers_num} workers for parallel processing")
            print(f"⚠️ Note: Multiple threads share the same model instance. Ensure GPU memory is sufficient.")
            
            def process_and_write(sample):
                """处理样本并立即写入（每条结果立即写入，确保数据安全）"""
                result = process_single_sample(sample)
                if result:
                    # 立即写入，不使用缓冲区
                    write_result(result, output_file, file_lock)
                return result
            
            with ThreadPoolExecutor(max_workers=workers_num) as executor:
                futures = [executor.submit(process_and_write, sample) for sample in samples_to_process]
                
                # 使用tqdm显示进度
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating (parallel)"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Future error: {e}")


def main():
    global SEARCH_ENGINE_URL, RAW_IMAGE_DIR, IMAGE_OUTPUT_DIR, MODEL_INPUT_IMAGE_DIR
    parser = argparse.ArgumentParser(description='Wiki-VISA multi-turn inference via an OpenAI-compatible endpoint')
    parser.add_argument('--data_file', type=str, required=True,
                       help='输入的parquet文件路径')
    parser.add_argument('--base_url', type=str, default="http://localhost:8001/v1",
                       help='vLLM API服务的基础URL（例如：http://localhost:8001/v1）')
    parser.add_argument('--api_key', type=str, default="EMPTY",
                       help='API密钥（vLLM服务可能不需要，但需要提供）')
    parser.add_argument('--model_name', type=str, default='mcite_model',
                       help='模型名称（如果为None则使用服务默认模型）')
    parser.add_argument('--output_file', type=str, default="outputs/wiki-visa/predictions.jsonl", help='输出结果文件路径（JSONL格式）')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0, help='生成温度')
    parser.add_argument('--max_iterations', type=int, default=5, help='最大迭代次数（用于多轮对话）')
    parser.add_argument('--workers_num', type=int, default=1, help='线程数（1=单线程，>1=多线程）')
    
    parser.add_argument('--search_url', default=SEARCH_ENGINE_URL)
    parser.add_argument('--image_root', default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument('--image_output_dir', default=None, help='Crop/debug directory; defaults beside output_file')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    if args.workers_num < 1 or args.max_iterations < 1:
        parser.error('--workers_num and --max_iterations must be positive')
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max_samples must be positive')
    SEARCH_ENGINE_URL = args.search_url
    args.data_file = str(project_path(args.data_file))
    if args.output_file:
        args.output_file = str(project_path(args.output_file))
    RAW_IMAGE_DIR = str(project_path(args.image_root))
    IMAGE_OUTPUT_DIR = str(project_path(args.image_output_dir)) if args.image_output_dir else str(project_path(args.output_file or "outputs/wiki-visa/predictions.jsonl").parent / "image_crop")
    MODEL_INPUT_IMAGE_DIR = str(Path(IMAGE_OUTPUT_DIR).parent / 'model_inputs')
    
    # 加载数据
    print(f"Loading data from: {args.data_file}")
    data = load_parquet_data(args.data_file, max_samples=args.max_samples)
    print(f"Loaded {len(data)} samples")
    
    # 初始化vLLM API客户端
    print(f"Initializing vLLM API client with base_url: {args.base_url}")
    print(f"Model name: {args.model_name or 'default'}")
    model = SFTModelInference(args.base_url, args.api_key, args.model_name)
    print("vLLM API client initialized successfully")
    
    # 设置输出文件
    if args.output_file is None:
        output_dir = str(PROJECT_ROOT / "outputs/wiki-visa")
        os.makedirs(output_dir, exist_ok=True)
        data_basename = os.path.splitext(os.path.basename(args.data_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = os.path.join(output_dir, f'inference_raw_{data_basename}_{timestamp}.jsonl')
    else:
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    
    print(f"Output file: {args.output_file}")
    
    # 批量生成
    model.batch_generate(
        data,
        output_file=args.output_file,
        workers_num=args.workers_num,
        max_iterations=args.max_iterations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print(f"\n✅ Inference completed! Results saved to: {args.output_file}")
    print(f"Each result contains 'all_responses' field with all raw model outputs.")


if __name__ == "__main__":
    main()

