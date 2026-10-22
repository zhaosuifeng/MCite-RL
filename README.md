# MCite-RL Startup Guide

Workflow: prepare Python environments and models → prepare data and retrieval indexes → start the retrieval API → start the model API for inference, or launch training directly.

Run all commands from the project root. Model weights, full datasets, and retrieval indexes must be prepared separately. `data/examples` contains format examples only.

## 1. Configure the Environments

Use one environment for retrieval and the inference client, and another for the vLLM model API and training.

| Purpose | Environment |
| --- | --- |
| Retrieval, data preparation, inference client, evaluation | Python 3.12; see `requirements/inference.lock.requirements.txt` |
| vLLM API, GRPO training | Python 3.11; validated combination: PyTorch 2.5.1/cu124, Transformers 4.49, vLLM 0.7.3, FlashAttention 2.7.4.post1 |

Point the launchers to existing Python environments, or create environments using an installed `uv`:

```bash
uv venv --python 3.11 /path/to/train-env
uv pip install --python /path/to/train-env/bin/python \
  --index-strategy unsafe-best-match -r requirements/train.lock.requirements.txt

uv venv --python 3.12 /path/to/infer-env
uv pip install --python /path/to/infer-env/bin/python \
  --index-strategy unsafe-best-match -r requirements/inference.lock.requirements.txt
```

The dependency snapshots come from a Linux GPU environment. FlashAttention, PyTorch, and CUDA versions must be compatible. When compiling CUDA extensions, set `CUDA_HOME` to the toolkit used by the corresponding environment.

Set these variables in each terminal, replacing `/path/to/...` with actual paths:

```bash
export TRAIN_PY=/path/to/train-env/bin/python
export INFER_PY=/path/to/infer-env/bin/python
export MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export EMBED_MODEL_PATH=/path/to/colqwen2-v0.1-merged

export TRAIN_FILE="$PWD/data/wiki-visa/train.parquet"
export VAL_FILE="$PWD/data/wiki-visa/val.parquet"
export TEST_FILE="$PWD/data/wiki-visa/test.parquet"
export TRAIN_IMAGES="$PWD/data/wiki-visa/train-images"
export TEST_IMAGES="$PWD/data/wiki-visa/test-images"
export TRAIN_INDEX="$PWD/corpus/wiki-visa/train"
export TEST_INDEX="$PWD/corpus/wiki-visa/test"

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

`MODEL_PATH` specifies the generation model; `EMBED_MODEL_PATH` specifies the retrieval encoder. Build each index with the same encoder used by its retrieval API.

## 2. Prepare the Data

### Using Prepared Data

Point `TRAIN_FILE`, `VAL_FILE`, and `TEST_FILE` to existing parquet files, and set the image and index directories accordingly. If matching indexes are already available, proceed to Section 4.

Training and validation parquet files must contain `prompt`, `data_source`, `reward_model`, and `extra_info`. See `data/examples/test.jsonl` for the inference format. Split training and validation data by page ID, and keep test data out of training.

### Preparing Raw Wiki-VISA Parquet Files

Download the dataset:

```bash
bash scripts/download_dataset.sh MrLight/wiki-visa \
  --dataset --local-dir wiki-visa
```

Create training and validation partitions from the original training shards and export their page images:

```bash
for partition in train val; do
  "$INFER_PY" scripts/prepare_wiki_visa.py \
    --input 'wiki-visa/data/train-*.parquet' \
    --format verl --partition "$partition" --validation_fraction 0.1 \
    --image_root "$TRAIN_IMAGES" \
    --output "data/wiki-visa/$partition.parquet"
done
```

Prepare the test data and images:

```bash
"$INFER_PY" scripts/prepare_wiki_visa.py \
  --input 'wiki-visa/data/test-*.parquet' \
  --image_root "$TEST_IMAGES" --output "$TEST_FILE"
```

The script stops if the output file already exists. Add `--overwrite` when you intend to replace it. To restrict samples to pages covered by an existing index, add `--index_dir /path/to/index`.

## 3. Build Retrieval Indexes

Skip this section if matching indexes are available. Index construction requires a GPU; run it before starting the generation model or training.

This function builds an index from the specified image directory, using one worker to limit concurrent GPU memory use:

```bash
build_index() {
  INDEX_IMAGES="$1" INDEX_DIR="$2" "$INFER_PY" - <<'PY'
import os
from search_engine.ingestion import Ingestion

engine = Ingestion(
    ".",
    input_prefix=os.environ["INDEX_IMAGES"],
    output_prefix=os.environ["INDEX_DIR"],
    embed_model_name=os.environ["EMBED_MODEL_PATH"],
)
engine.workers = 1
engine.ingestion_multi_session()
PY
}

build_index "$TRAIN_IMAGES" "$TRAIN_INDEX"
build_index "$TEST_IMAGES" "$TEST_INDEX"
```

Each index directory should contain `.node` files with names matching the page images. When changing the encoder, rebuild into a new directory to avoid mixing incompatible vectors.

## 4. Start the Retrieval API

Start the service needed for your task. On a single GPU with limited free memory, keep only the required retrieval service running.

Inference retrieval API, port `8002`:

```bash
"$INFER_PY" -m search_engine.search_engine_api \
  --index_dir "$TEST_INDEX" --image_root "$TEST_IMAGES" \
  --embed_model_name "$EMBED_MODEL_PATH" \
  --host 127.0.0.1 --port 8002
```

Training retrieval API, port `8003`:

```bash
"$INFER_PY" -m search_engine.search_engine_api \
  --index_dir "$TRAIN_INDEX" --image_root "$TRAIN_IMAGES" \
  --embed_model_name "$EMBED_MODEL_PATH" \
  --host 127.0.0.1 --port 8003
```

Check the service from another terminal. Wait for `status: ok` before proceeding:

```bash
curl --noproxy '*' http://127.0.0.1:8002/health
# For the training retrieval service, check http://127.0.0.1:8003/health
```

## 5. Start the Model API and Run Inference

### Start the Generation Model

Keep the inference retrieval service running and start the model API in another terminal:

```bash
MODEL_PATH="$MODEL_PATH" bash scripts/wiki_visa.sh model \
  --gpu-memory-utilization 0.55
```

This command uses vLLM from `TRAIN_PY`, listens on port `8001`, and serves the model as `vrag_model`. Check readiness:

```bash
curl --noproxy '*' http://127.0.0.1:8001/v1/models
```

### Run Multi-Turn Inference

Start with five samples:

```bash
"$INFER_PY" scripts/inference_wiki_visa.py \
  --data_file "$TEST_FILE" \
  --base_url http://127.0.0.1:8001/v1 --model_name vrag_model \
  --search_url http://127.0.0.1:8002/search \
  --image_root "$TEST_IMAGES" \
  --output_file outputs/wiki-visa/predictions.jsonl \
  --max_samples 5 --max_iterations 5 \
  --max_new_tokens 2048 --temperature 0 --workers_num 1
```

Remove `--max_samples 5` to process the full dataset. Existing output is resumed by UID; use a new output filename to rerun samples.

To use an existing model service, set `--base_url` and `--model_name` to its endpoint and served model name. No additional model API process is needed.

### Evaluate Predictions

```bash
"$INFER_PY" scripts/evaluate_wiki_visa.py \
  --inference_file outputs/wiki-visa/predictions.jsonl \
  --reference_image_root "$TEST_IMAGES" \
  --wiki_visa_parquet_root "$(dirname "$TEST_FILE")" \
  --output_file outputs/wiki-visa/metrics.json
```

## 6. Launch Training

Training loads the actor, reference model, and vLLM rollout internally; it does not require a separate model API. On a single GPU, stop the model API from Section 5 and any unused inference retrieval service. Keep the training retrieval API on port `8003` running.

### Inspect the Configuration

```bash
bash mytrain.sh --cfg job
```

### Run One Training Step

```bash
MODEL_PATH="$MODEL_PATH" \
TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
SEARCH_URL=http://127.0.0.1:8003/search \
bash mytrain.sh
```

The default single-GPU configuration uses one question per batch, two agent trajectories, micro batch size 1, generation concurrency 1, up to 256 generated tokens per call, and two interaction rounds. Gradient checkpointing and parameter/optimizer offloading are enabled. By default, training performs **one optimizer update** and saves a checkpoint to verify the pipeline.

Checkpoints are written to `checkpoints/vrag/<reward_mode>_<timestamp>/` by default.

### Adjust Training Duration and Interaction Rounds

```bash
TOTAL_TRAINING_STEPS=100 TOTAL_EPOCHS=2 MAX_TURNS=5 \
SEARCH_URL=http://127.0.0.1:8003/search bash mytrain.sh
```

Training stops when either the step limit or epoch limit is reached. Set both according to the dataset size. Use environment variables such as `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `N_AGENT`, and `MAX_RESPONSE_LENGTH` to adjust parameters, or append Hydra overrides to the command.

### Resume Training

Resume from a specific saved step:

```bash
RESUME_FROM_STEP_DIR=/path/to/run/global_step_100 \
SEARCH_URL=http://127.0.0.1:8003/search bash mytrain.sh
```

Or resume from the latest checkpoint in a run directory:

```bash
RESUME_CHECKPOINT_DIR=/path/to/run \
SEARCH_URL=http://127.0.0.1:8003/search bash mytrain.sh
```

Keep the model and training configuration compatible with the checkpoint. Set `TOTAL_TRAINING_STEPS` to the target total step count. Old checkpoints are not loaded automatically by default.

## 7. Prompt and Reward

Training and inference load the same first-user instruction from `prompts/wiki_visa_sft_gemini.txt`. When loading existing Wiki-VISA parquet files, the instruction is refreshed in memory using the question. Normal retrieval and crop observations contain only the image.

The default reward mode in `mytrain.sh` is `search_answer_citation_bbox_uniform`:

```text
reward = 0.40 × answer Exact Match
       + 0.35 × citation IoU
       + 0.15 × retrieval NDCG
       + 0.10 × citation IoGT
```

The original format requirement is retained: search, bbox, answer, and citation tags must all be present, or the total reward is zero. IoGT is the intersection of the citation and ground-truth boxes divided by the ground-truth box area. KL loss is computed separately, with a default coefficient of `0.001`.

To select another original reward mode, such as answer EM only:

```bash
REWARD_COMBINATION_MODE=search-r1 bash mytrain.sh
```

## 8. Use Example Data and Run Tests

`data/examples` contains two training rows, one validation row, two test rows, and five images. Inspect the JSONL files to see the data format.

Inspect the training configuration with example data:

```bash
TRAIN_FILE=data/examples/train.parquet \
VAL_FILE=data/examples/val.parquet \
bash mytrain.sh --cfg job
```

The examples do not include a retrieval index. Actual training or inference still requires a matching index and model.

If the `tests` directory is present, run the CPU checks in the training environment:

```bash
"$TRAIN_PY" -m pytest tests -q
```
