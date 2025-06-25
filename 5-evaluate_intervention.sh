#!/bin/bash

DATA_DIR="data"
RESPONSES_DIR="intervention/k128_alpha20_target1"

python evaluate_intervene.py \
    --responses_dir "$RESPONSES_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$RESPONSES_DIR"