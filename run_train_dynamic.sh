#!/bin/bash

# Dynamic Training Script for IM2ELEVATION
# Usage: ./run_train_dynamic.sh <dataset_dir> <input_size> [epochs] [batch_size] [lr]

set -e

# Default values
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=1
DEFAULT_LR=0.0001

# Parse arguments
if [ $# -lt 2 ]; then
    echo "❌ Usage: $0 <dataset_dir> <input_size> [epochs] [batch_size] [lr]"
    echo "   Examples:"
    echo "     $0 DFC2023Amini 440                    # Standard training"
    echo "     $0 DFC2023Amini 512 50 2 0.0005       # Custom parameters"
    echo "     $0 DFC2023Amini 256 100 4 0.0001      # Smaller input size"
    exit 1
fi

DATASET_DIR="$1"
INPUT_SIZE="$2"
EPOCHS="${3:-$DEFAULT_EPOCHS}"
BATCH_SIZE="${4:-$DEFAULT_BATCH_SIZE}"
LR="${5:-$DEFAULT_LR}"

# Validate input size
if ! [[ "$INPUT_SIZE" =~ ^[0-9]+$ ]] || [ "$INPUT_SIZE" -lt 128 ] || [ "$INPUT_SIZE" -gt 1024 ]; then
    echo "❌ Invalid input size: $INPUT_SIZE (must be 128-1024)"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Look for CSV file
CSV_FILE=""
if [ -f "$DATASET_DIR/train_valid.csv" ]; then
    CSV_FILE="$DATASET_DIR/train_valid.csv"
elif [ -f "$DATASET_DIR/train.csv" ]; then
    CSV_FILE="$DATASET_DIR/train.csv"
else
    echo "❌ No training CSV found in $DATASET_DIR"
    echo "   Expected: train_valid.csv or train.csv"
    exit 1
fi

echo "🚀 IM2ELEVATION Dynamic Training"
echo "================================="
echo "📁 Dataset: $DATASET_DIR"
echo "📊 CSV: $(basename $CSV_FILE)"
echo "📐 Input Size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "📦 Batch Size: $BATCH_SIZE"
echo "🔢 Epochs: $EPOCHS"
echo "📈 Learning Rate: $LR"
echo ""

# Memory estimation
MEMORY_GB=$((INPUT_SIZE * INPUT_SIZE * BATCH_SIZE * 4 / 1000000))
echo "💾 Estimated GPU Memory: ~${MEMORY_GB}GB (batch size $BATCH_SIZE)"

if [ "$MEMORY_GB" -gt 8 ]; then
    echo "⚠️  High memory usage detected. Consider reducing batch size or input size."
fi

echo ""
echo "▶️  Starting training..."
echo ""

# Start training with dynamic parameters
python train_dynamic.py \
    --data "$DATASET_DIR" \
    --csv "$CSV_FILE" \
    --input_size "$INPUT_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR"

echo ""
echo "✅ Dynamic training completed!"
echo "📁 Models saved in: $DATASET_DIR/"
echo "   - ${DATASET_DIR##*/}_dynamic_${INPUT_SIZE}_model_best.pkl"
echo "   - ${DATASET_DIR##*/}_dynamic_${INPUT_SIZE}_model_latest.pkl"
