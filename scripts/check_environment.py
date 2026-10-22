#!/usr/bin/env python3
"""Exercise installed GPU runtimes without downloading model weights or data."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--role', choices=('train', 'inference'), required=True)
parser.add_argument('--output')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
report = {'role': args.role, 'python': sys.version.split()[0], 'checks': {}, 'errors': {}}

def check(name, function):
    try:
        result = function()
        report['checks'][name] = result if result is not None else True
        print(f'PASS {name}', flush=True)
    except Exception:
        report['errors'][name] = traceback.format_exc()
        print(f'FAIL {name}: {report["errors"][name]}', flush=True)

def imports():
    names = (['verl', 'vllm', 'ray', 'tensordict', 'torchdata.stateful_dataloader',
              'hydra', 'qwen_vl_utils', 'verl.trainer.main_ppo'] if args.role == 'train'
             else ['search_engine.search_engine_api', 'search_engine.vl_embedding',
                   'colpali_engine.models', 'pyarrow.parquet', 'openai', 'fastapi', 'PIL'])
    for name in names:
        importlib.import_module(name)
    return names

check('runtime_imports', imports)
import torch
report['torch'] = torch.__version__
report['cuda_runtime'] = torch.version.cuda
report['gpu'] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
report['versions'] = {}
for package in ('torchvision', 'transformers', 'colpali-engine', 'vllm', 'flash-attn', 'ray'):
    try:
        report['versions'][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        pass

def cuda_compute():
    assert torch.cuda.is_available(), 'No visible CUDA GPU'
    x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = x @ x
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    return {'bf16': torch.cuda.is_bf16_supported(), 'capability': torch.cuda.get_device_capability()}

check('cuda_bf16_forward_backward', cuda_compute)

def vision():
    from torchvision.ops import nms
    boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 1, 1]], device='cuda', dtype=torch.float32)
    result = nms(boxes, torch.tensor([0.9, 0.8], device='cuda'), 0.5)
    assert result.numel() == 2
    return True

check('torchvision_cuda_extension', vision)

if args.role == 'train':
    def flash():
        from flash_attn import flash_attn_func
        q = torch.randn(2, 128, 4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        result = flash_attn_func(q, k, v, causal=True)
        # Only the correctness reference uses PyTorch; the operation under test is FA2.
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        error = (result.float() - reference.float()).abs().max().item()
        assert error < 0.05, error
        result.float().square().mean().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(q.grad).all()
        return {'max_abs_error': error, 'forward_backward': True}

    check('flashattention2_cuda', flash)

    def vllm_flash():
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        q = torch.randn(32, 4, 64, device='cuda', dtype=torch.bfloat16)
        cu = torch.tensor([0, 16, 32], device='cuda', dtype=torch.int32)
        result = flash_attn_varlen_func(q=q, k=q, v=q, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                       max_seqlen_q=16, max_seqlen_k=16, causal=True)
        if isinstance(result, tuple):
            result = result[0]
        torch.cuda.synchronize()
        assert result.shape == q.shape and torch.isfinite(result).all()
        return True

    check('vllm_flashattention_cuda', vllm_flash)

    def nccl():
        import torch.distributed as dist
        with tempfile.TemporaryDirectory(prefix='mcite-nccl-') as folder:
            dist.init_process_group('nccl', init_method=f'file://{folder}/store', rank=0, world_size=1)
            try:
                value = torch.ones(1, device='cuda')
                dist.all_reduce(value)
                torch.cuda.synchronize()
                assert value.item() == 1
            finally:
                dist.destroy_process_group()
        return True

    check('nccl_single_gpu', nccl)

def tiny_qwen():
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    options = dict(vocab_size=128, hidden_size=128, intermediate_size=256,
                   num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                   image_token_id=125, video_token_id=126, vision_start_token_id=127,
                   vision_end_token_id=124, rope_scaling={'type': 'mrope', 'mrope_section': [4, 6, 6]},
                   vision_config=dict(depth=1, embed_dim=64, hidden_size=128, num_heads=4,
                                      patch_size=14, temporal_patch_size=2, spatial_merge_size=2, mlp_ratio=2))
    if args.role == 'inference':
        # Transformers 5 separates the text model's config.
        text_keys = ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                     'num_attention_heads', 'num_key_value_heads', 'rope_scaling')
        options['text_config'] = {key: options[key] for key in text_keys}
    config = Qwen2VLConfig(**options)
    if args.role == 'train':
        config._attn_implementation = 'flash_attention_2'
        model = Qwen2VLForConditionalGeneration(config).to(device='cuda', dtype=torch.bfloat16)
    else:
        from colpali_engine.models import ColQwen2
        model = ColQwen2(config).to(device='cuda', dtype=torch.bfloat16)
    tokens = torch.tensor([[1, 2, 3, 4]], device='cuda')
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        result = model(input_ids=tokens, attention_mask=mask)
    values = result.logits if args.role == 'train' else result
    torch.cuda.synchronize()
    assert torch.isfinite(values).all()
    image_tokens = torch.tensor([[1, 127, 125, 124, 2]], device='cuda')
    pixels = torch.randn(4, 3 * 2 * 14 * 14, device='cuda', dtype=torch.bfloat16)
    if args.role == 'inference':
        pixels = pixels.unsqueeze(0)
    with torch.no_grad():
        visual_result = model(input_ids=image_tokens, attention_mask=torch.ones_like(image_tokens),
                              pixel_values=pixels,
                              image_grid_thw=torch.tensor([[1, 2, 2]], device='cuda'))
    visual_values = visual_result.logits if args.role == 'train' else visual_result
    torch.cuda.synchronize()
    assert torch.isfinite(visual_values).all()
    return {'text_shape': list(values.shape), 'image_shape': list(visual_values.shape), 'random_weights': True,
            'purpose': 'library compatibility only, not model quality'}

check('tiny_visual_model_text_and_image_forward', tiny_qwen)
report['ready'] = not report['errors']
text = json.dumps(report, ensure_ascii=False, indent=2)
print(text, flush=True)
if args.output:
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + '\n')
raise SystemExit(0 if report['ready'] else 1)
