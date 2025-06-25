#!/bin/bash

MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-70B-Instruct" "CohereLabs/aya-expanse-8b"  "CohereLabs/aya-expanse-32b" "mistralai/Ministral-8B-Instruct-2410" "mistralai/Mistral-Small-3.2-24B-Instruct-2506")

DATA_DIR="data"

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/:' '-')

  RESPONSES_DIR="responses/responses_${SAFE_MODEL_NAME}"
  OUTPUT_DIR="analysis/analysis_${SAFE_MODEL_NAME}"

  echo "Evaluating model: $MODEL"

  python evaluate.py \
    --responses_dir "$RESPONSES_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"
done
