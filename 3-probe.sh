#!/bin/bash

pip install git+https://github.com/davidbau/baukit
pip install -U torch accelerate transformers numba
pip install numpy==1.26.4

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

models=("Qwen/Qwen3-32B"
        # "Qwen/Qwen3-8B" 
        # "Qwen/Qwen3-14B" 
        # "meta-llama/Llama-3.1-8B-Instruct" 
        # "meta-llama/Llama-3.1-70B-Instruct"
        # "CohereLabs/aya-expanse-8b"
        "CohereLabs/aya-expanse-32b" 
        "mistralai/Ministral-8B-Instruct-2410")



for model in "${models[@]}"; do
    echo "Running probes for model: $model"
    python probe_binary.py \
        --model_name "$model" \
        --max_samples 2000 \
        --seed 42 \
        --binary
done

