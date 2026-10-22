#!/usr/bin/env python3
"""Run the existing VERL trainer in a bounded local Ray cluster."""
import os
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['ENSURE_CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
os.environ.setdefault('REWARD_COMBINATION_MODE', 'search_answer_citation_bbox_uniform')
os.environ.setdefault('REFUSAL_JUDGE_ENABLED', '0')
os.environ.setdefault('REFUSAL_METRICS_ENABLED', '0')
os.environ.setdefault('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['PYTHONPATH'] = str(ROOT) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')

import ray

def run_trainer():
    # Execute as the original entry point so Hydra resolves its file-based config directory.
    runpy.run_module('verl.trainer.main_ppo', run_name='__main__')

if __name__ == '__main__':
    if any(arg in sys.argv for arg in ('--cfg', '--help', '-h')):
        run_trainer()
    else:
        names = ('CUDA_HOME', 'PYTHONPATH', 'REWARD_COMBINATION_MODE', 'REFUSAL_JUDGE_ENABLED',
                 'REFUSAL_METRICS_ENABLED', 'VLLM_ATTENTION_BACKEND', 'HF_HUB_OFFLINE',
                 'TRANSFORMERS_OFFLINE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'ENSURE_CUDA_VISIBLE_DEVICES', 'NO_PROXY', 'no_proxy',
                 'PYTORCH_CUDA_ALLOC_CONF')
        env = {key: os.environ[key] for key in names if key in os.environ}
        env.update(TOKENIZERS_PARALLELISM='false', NCCL_DEBUG='WARN', VLLM_LOGGING_LEVEL='INFO')
        ray.init(address='local', num_cpus=4, num_gpus=1, object_store_memory=1024**3,
                 include_dashboard=False, _temp_dir='/tmp/mcite-ray-qwen3b',
                 runtime_env={'env_vars': env})
        try:
            run_trainer()
        finally:
            ray.shutdown()
