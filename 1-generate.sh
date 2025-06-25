#!/bin/bash

pip install vllm torch transformers

HF_TOKEN=*

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-70B-Instruct" "CohereLabs/aya-expanse-8b"  "CohereLabs/aya-expanse-32b" "mistralai/Ministral-8B-Instruct-2410" "mistralai/Mistral-Small-3.2-24B-Instruct-2506")

PROMPT_DIR="./prompts"
QUESTIONS_DIR="./data"
SEED=42
TEMPERATURE=0.7
TOP_P=0.9
MAX_TOKENS=256

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/:' '-')

  OUTPUT_DIR="responses_${SAFE_MODEL_NAME}/"

  echo "Running for model: $MODEL"
  
  python generate.py \
    --model "$MODEL" \
    --prompts_dir "$PROMPT_DIR" \
    --questions_dir "$QUESTIONS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS"
done
