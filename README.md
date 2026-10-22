# MCite-RL

Run the commands below from the project root. Prepare your model weights, dataset, page images, and matching retrieval index separately. Dataset paths are configurable, for example for Wiki-VISA or FinRAGBench.

## 1. Environment Setup

Use separate environments for retrieval/inference and training/vLLM:

```bash
uv venv --python 3.12 /path/to/infer-env
uv pip install --python /path/to/infer-env/bin/python \
  --index-strategy unsafe-best-match -r requirements/inference.lock.requirements.txt

uv venv --python 3.11 /path/to/train-env
uv pip install --python /path/to/train-env/bin/python \
  --index-strategy unsafe-best-match -r requirements/train.lock.requirements.txt

export INFER_PY=/path/to/infer-env/bin/python
export TRAIN_PY=/path/to/train-env/bin/python
export MODEL_PATH=/path/to/policy-model
export EMBED_MODEL_PATH=/path/to/colqwen2-v0.1-merged
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

The validated training stack uses PyTorch 2.5.1/cu124, vLLM 0.7.3, and FlashAttention 2.7.4.post1. Use compatible CUDA and FlashAttention builds.

## 2. Deploy the Retrieval API

Use an index built with the same retrieval encoder:

```bash
"$INFER_PY" -m search_engine.search_engine_api \
  --index_dir /path/to/index --image_root /path/to/images \
  --embed_model_name "$EMBED_MODEL_PATH" \
  --host 127.0.0.1 --port 8002
```

Index builders are available in `search_engine/ingestion.py` and `search_engine/ingestion_finragbench.py`. Use the appropriate corpus for training or evaluation.

## 3. Run Inference

Start the model API in a separate terminal with the environment variables above:

```bash
bash scripts/wiki_visa.sh model --gpu-memory-utilization 0.55
```

Then run multi-turn inference:

```bash
"$INFER_PY" scripts/inference_wiki_visa.py \
  --data_file /path/to/test.parquet \
  --base_url http://127.0.0.1:8001/v1 --model_name vrag_model \
  --search_url http://127.0.0.1:8002/search \
  --image_root /path/to/images \
  --output_file outputs/predictions.jsonl \
  --max_samples 5 --max_iterations 5 --temperature 0
```

Remove `--max_samples 5` to process all samples. Normalize your data to the format shown in `data/examples`; register reward handling for new training task identifiers as needed.

## 4. Run Training

Keep a retrieval API serving the training corpus running. Training loads its own generation models; stop the standalone model API first when sharing one GPU.

```bash
MODEL_PATH=/path/to/policy-model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
SEARCH_URL=http://127.0.0.1:8002/search \
bash mytrain.sh
```

The default is a single-GPU, one-step smoke run. Set `TOTAL_TRAINING_STEPS` and `TOTAL_EPOCHS` for longer training, or append `--cfg job` to inspect the configuration.

The default reward is `0.40 × EM + 0.35 × IoU + 0.15 × NDCG + 0.10 × IoGT`. The original mode requires search, bbox, answer, and citation tags. The shared prompt is in `prompts/wiki_visa_sft_gemini.txt`.

## Acknowledgments

Our implementation references [VRAG-RL](https://github.com/Alibaba-NLP/VRAG/tree/main/VRAG-RL) from Alibaba-NLP/VRAG. We thank the authors for making their code available.
