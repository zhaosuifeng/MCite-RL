import re
import os
import base64
import requests
import io
import ast
from PIL import Image, ImageDraw
from verl.utils.reward_score import vrag

def extract_bbox_and_think(response_str):
    """
    Extract <bbox>[x1, y1, x2, y2]</bbox> and <think>content</think> from response string.
    """
    bbox = None
    think_content = None

    # Extract bbox
    bbox_match = re.search(r'<bbox>\s*\[([^\]]+)\]\s*</bbox>', response_str)
    if bbox_match:
        try:
            bbox_list = ast.literal_eval(f"[{bbox_match.group(1)}]")
            if isinstance(bbox_list, list) and len(bbox_list) == 4:
                bbox = bbox_list
        except:
            pass

    # Extract think
    think_match = re.search(r'<think>(.*?)</think>', response_str, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()

    return bbox, think_content

def draw_bbox_on_image(image_path, bbox):
    """
    Load image and draw a red bounding box.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        # Draw red rectangle with width 3
        draw.rectangle(bbox, outline="red", width=5)
        return img
    except Exception as e:
        print(f"[Relevance] Error drawing bbox: {e}")
        return None

def encode_image_to_base64(image):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"[Relevance] Error encoding image: {e}")
        return None

def call_rm_for_relevance(image_base64, think_content, rm_config):
    """
    Call the Reward Model API to evaluate relevance.
    """
    url = rm_config.get('rm_url')
    key = rm_config.get('rm_key')
    model = rm_config.get('rm_model_name')
    
    prompt = f"""You are an expert evaluator. 
Please evaluate the relevance between the marked region (red box) in the image and the following thought process:

<think>
{think_content}
</think>

Does the thought process accurately describe, reason about, or attend to the information contained within the marked region?

Scoring Criteria (0-5):
- 0: Completely irrelevant or hallucinated. The thought process describes something not present in the region at all.
- 1: Very low relevance. Mentions a minor detail but largely misses the main content of the region, or contains significant errors.
- 2: Low relevance. Captures some correct visual elements but misses key information or makes wrong inferences.
- 3: Moderate relevance. Correctly identifies the main subject of the region but lacks depth or specific details.
- 4: High relevance. Accurately describes the region and its contents with good detail and correct reasoning.
- 5: Perfect relevance. The thought process perfectly aligns with the visual content of the marked region, capturing fine-grained details and showing deep understanding.

Return ONLY a single integer score between 0 and 5.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    # Adapt for different APIs if necessary. Assuming OpenAI/vLLM chat completion format.
    # Note: Some APIs expect "image_url" with base64.
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        
        # Parse content for a score
        content = res_json['choices'][0]['message']['content']
        print("model rated relevance score:")
        print(content)
        # Try to find a number in the content
        # E.g. "5", "Score: 4"
        score_match = re.search(r"(\d+)", content)
        if score_match:
            score = float(score_match.group(1))
            # Normalize to 0-1 range for the reward system
            normalized_score = min(max(score, 0.0), 5.0) / 5.0
            return normalized_score
        else:
            return 0.0
    except Exception as e:
        print(f"[Relevance] RM API Error: {e}")
        return 0.0

def evaluate_relevance(response_str, extra_info, rm_config):
    """
    Main function to evaluate relevance.
    """
    bbox, think_content = extract_bbox_and_think(response_str)
    
    if not bbox or not think_content:
        # If components missing, return 0.0
        print(f"[Relevance Debug] Missing bbox or think_content. bbox={bbox}, think_content_len={len(think_content) if think_content else 0}")
        return 0.0

    retrievaled_images = extra_info.get('retrievaled_images', [])
    crop_history = extra_info.get('crop_history', [])
    
    # Debug info
    print(f"[Relevance Debug] retrievaled_images count: {len(retrievaled_images)}")
    # print(f"[Relevance Debug] crop_history: {crop_history}")

    # Use vrag utils to map bbox to original image
    try:
        crop_chain = vrag.build_crop_chain_from_extra_info(retrievaled_images, crop_history)
        
        # Identify the target image (last search result in the chain that is valid)
        # The crop chain logic maps coordinates to the 'last_search_step'.
        search_steps = [step for step in crop_chain if step['type'] == 'search']
        if not search_steps:
            print("[Relevance Debug] No search steps found in crop chain.")
            return 0.0
        
        last_search_step = search_steps[-1]
        original_image_path = last_search_step['result_image']
        print(f"[Relevance Debug] Original image path: {original_image_path}")
        
        # Convert bbox
        # NOTE: vrag.convert_crop_coords_to_search_image expects 'citation_coords' but we pass 'bbox'.
        # The logic is the same: mapping from current view to original view.
        original_bbox = vrag.convert_crop_coords_to_search_image(bbox, crop_chain)
        print(f"[Relevance Debug] Converted original_bbox: {original_bbox}")
        
        if not original_bbox:
            print("[Relevance Debug] Failed to convert bbox to original image coords.")
            return 0.0

        # Draw on image
        img_with_box = draw_bbox_on_image(original_image_path, original_bbox)
        if not img_with_box:
            print(f"[Relevance Debug] Failed to draw bbox on image: {original_image_path}")
            return 0.0
            
        # Encode
        img_b64 = encode_image_to_base64(img_with_box)
        if not img_b64:
            print("[Relevance Debug] Failed to encode image to base64.")
            return 0.0
            
        # Call RM
        print("[Relevance Debug] Calling RM API...")
        score = call_rm_for_relevance(img_b64, think_content, rm_config)
        return score

    except Exception as e:
        print(f"[Relevance] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

