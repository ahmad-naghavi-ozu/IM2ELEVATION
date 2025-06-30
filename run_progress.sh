#!/bin/bash

# Simple Training Progress Checker
# Usage: ./progress.sh [DATASET_NAME] [OUTPUT_DIR]

DATASET_NAME=${1:-"DFC2023S"}
OUTPUT_DIR=${2:-"pipeline_output"}
DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"

if [[ ! -d "$DATASET_OUTPUT_DIR" ]]; then
    echo "❌ No training directory found: $DATASET_OUTPUT_DIR"
    exit 1
fi

echo "📊 Training Progress for $DATASET_NAME"
echo "========================================"

# Count checkpoints
CHECKPOINTS=$(find "$DATASET_OUTPUT_DIR" -name "*.tar" 2>/dev/null | wc -l)
echo "💾 Checkpoints saved: $CHECKPOINTS"

# Show best checkpoint
BEST_FILE=$(find "$DATASET_OUTPUT_DIR" -name "*best_epoch_*.tar" 2>/dev/null | head -1)
if [[ -n "$BEST_FILE" ]]; then
    BEST_EPOCH=$(echo "$BEST_FILE" | grep -o 'epoch_[0-9]*' | cut -d'_' -f2)
    echo "🏆 Best epoch: $BEST_EPOCH"
fi

# Show directory size
DIR_SIZE=$(du -sh "$DATASET_OUTPUT_DIR" 2>/dev/null | cut -f1)
echo "📁 Directory size: $DIR_SIZE"

echo "========================================"
