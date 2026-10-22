#!/usr/bin/env bash
# Wiki-VISA commands; all filesystem arguments are relative to the project root.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
INFER_PY="${INFER_PY:-python}"
TRAIN_PY="${TRAIN_PY:-python}"
MODEL_PATH="${MODEL_PATH:-models/policy}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-models/colqwen2-v0.1-merged}"
RETRIEVAL_INDEX="${RETRIEVAL_INDEX:-corpus/wiki-visa/val_batch_0}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8002/search}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8001/v1}"
command_name="${1:-help}"
[[ $# -eq 0 ]] || shift
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
run() {
    printf '  '; printf '%q ' "$@"; printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}
case "$command_name" in
    prepare)
        prepare_args=(--input 'wiki-visa/data/test-*.parquet' --skip_incomplete \
            --image_root data/wiki-visa/images --output data/wiki-visa/test.parquet)
        prepare_index="${PREPARE_INDEX_DIR:-$RETRIEVAL_INDEX}"
        if [[ "$prepare_index" != none ]]; then prepare_args+=(--index_dir "$prepare_index"); fi
        run "$INFER_PY" scripts/prepare_wiki_visa.py "${prepare_args[@]}" "$@"
        ;;
    prepare-train)
        # Both partitions come only from HF train, split by page ID. Test is never used for training.
        for partition in train val; do
            run "$INFER_PY" scripts/prepare_wiki_visa.py \
                --input "${TRAIN_INPUT:-wiki-visa/data/train-*.parquet}" \
                --index_dir 'corpus/wiki-visa/train_batch_*' \
                --image_root data/wiki-visa/images --format verl \
                --partition "$partition" --validation_fraction 0.1 \
                --output "data/wiki-visa/$partition.parquet" "$@"
        done
        run "$INFER_PY" scripts/prepare_wiki_visa_index.py \
            --source 'corpus/wiki-visa/train_batch_*' \
            --image_root data/wiki-visa/images --output corpus/wiki-visa/train_available
        ;;
    check)
        run "$INFER_PY" scripts/check_wiki_visa.py --stage data "$@"
        run "$TRAIN_PY" scripts/check_wiki_visa.py --stage train --report_only
        ;;
    ingest)
        # Rebuild into a new directory if the original index encoder is unavailable.
        rebuilt="${REBUILT_INDEX:-corpus/wiki-visa/test_rebuilt}"
        if [[ "$DRY_RUN" == 0 ]] && [[ -d "$rebuilt" ]] && compgen -G "$rebuilt/*.node" >/dev/null; then
            printf 'Index already exists: %s; select a new REBUILT_INDEX.\n' "$rebuilt" >&2; exit 2
        fi
        run "$INFER_PY" -c 'import sys; from search_engine.ingestion import Ingestion; e=Ingestion(".", input_prefix="data/wiki-visa/images", output_prefix=sys.argv[1], embed_model_name=sys.argv[2]); e.workers=1; e.ingestion_multi_session()' "$rebuilt" "$EMBED_MODEL_PATH"
        ;;
    search)
        run "$INFER_PY" scripts/check_wiki_visa.py --stage search --index_dir "$RETRIEVAL_INDEX" --embed_model "$EMBED_MODEL_PATH"
        run "$INFER_PY" -m search_engine.search_engine_api \
            --index_dir "$RETRIEVAL_INDEX" --image_root data/wiki-visa/images \
            --embed_model_name "$EMBED_MODEL_PATH" --host 127.0.0.1 --port 8002 "$@"
        ;;
    model)
        export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
        run "$TRAIN_PY" scripts/check_wiki_visa.py --stage model --model "$MODEL_PATH"
        # vLLM is installed in agent-train, not agent-inference. Keep it in this environment.
        run "$TRAIN_PY" -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" --served-model-name vrag_model \
            --host 127.0.0.1 --port 8001 --tensor-parallel-size 1 \
            --dtype bfloat16 --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.70 \
            --limit-mm-per-prompt 'image=8' --enforce-eager "$@"
        ;;
    infer)
        run "$INFER_PY" scripts/inference_wiki_visa.py \
            --data_file data/wiki-visa/test.parquet --base_url "$MODEL_URL" \
            --model_name vrag_model --search_url "$SEARCH_URL" \
            --image_root data/wiki-visa/images \
            --output_file outputs/wiki-visa/predictions.jsonl \
            --max_iterations 5 --workers_num 1 "$@"
        ;;
    train|train-config)
        # Local rule reward: NDCG .15 + answer EM .40 + citation IoU .35 + IoGT .10.
        export REWARD_COMBINATION_MODE="${REWARD_COMBINATION_MODE:-search_answer_citation_bbox_uniform}"
        export REFUSAL_JUDGE_ENABLED="${REFUSAL_JUDGE_ENABLED:-0}"
        export REFUSAL_METRICS_ENABLED="${REFUSAL_METRICS_ENABLED:-0}"
        export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
        if [[ "$command_name" == train ]]; then
            run "$TRAIN_PY" scripts/check_wiki_visa.py --stage train --model "$MODEL_PATH"
        fi
        extra_args=()
        if [[ "$command_name" == train-config ]]; then extra_args+=(--cfg job); fi
        run "$TRAIN_PY" -m verl.trainer.main_ppo --config-name wiki_visa_grpo \
            "actor_rollout_ref.model.path=$MODEL_PATH" "retriever.url=$SEARCH_URL" \
            "${extra_args[@]}" "$@"
        ;;
    eval)
        run "$INFER_PY" scripts/evaluate_wiki_visa.py \
            --inference_file outputs/wiki-visa/predictions.jsonl \
            --reference_image_root data/wiki-visa/images \
            --wiki_visa_parquet_root data/wiki-visa \
            --output_file outputs/wiki-visa/metrics.json "$@"
        ;;
    *)
        cat <<'EOF'
Usage: bash scripts/wiki_visa.sh COMMAND [--dry-run] [extra arguments]
  prepare          Export pages and inference rows matching the existing validation index
  prepare-train    Convert downloaded HF train shards to disjoint VERL train/val partitions
  check            Check extracted data and report training prerequisites
  ingest           Rebuild vectors with a chosen encoder into a NEW index directory
  search           Start retrieval API on 8002 (requires matching encoder weights + GPU)
  model            Start vLLM model API on 8001 (uses agent-train environment)
  infer            Run multi-turn inference; append --max_samples 5 for a smoke run
  train-config     Compose and print training configuration without GPUs or model loading
  train            Launch one GRPO integration step; requires train data, weights and GPU
  eval             Evaluate saved predictions
Environment overrides: MODEL_PATH, EMBED_MODEL_PATH, RETRIEVAL_INDEX, SEARCH_URL, MODEL_URL.
Run --dry-run immediately after COMMAND to print commands without starting anything.
EOF
        ;;
esac
