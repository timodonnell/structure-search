#!/bin/bash
set -e

# Structure Prediction Training Script
# Trains Llama 3.1 8B on Foldseek data for sequence-to-structure prediction

cd /home/ubuntu/structure-search

# Configuration
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
DB_PATH="${DB_PATH:-data/foldseek/afdb50/afdb50}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/structure_predictor}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"

echo "============================================"
echo "Structure Prediction Training"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Database: $DB_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * 8))"
echo "Max sequence length: $MAX_LENGTH"
echo "Learning rate: $LEARNING_RATE"
echo "============================================"

# Launch training with accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --model-name "$MODEL_NAME" \
    --db-path "$DB_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --max-length "$MAX_LENGTH" \
    --learning-rate "$LEARNING_RATE" \
    --num-epochs "$NUM_EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 500
