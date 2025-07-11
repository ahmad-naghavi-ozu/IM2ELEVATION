#!/bin/bash

# IM2ELEVATION Training Script
# Usage: ./train_model.sh [OPTIONS]

set -e  # Exit on any error

# Default values
DATASET_NAME="DFC2023Amini"
EPOCHS=100
LEARNING_RATE=0.0001
OUTPUT_DIR="models_output"
CSV_PATH=""
RESUME_EPOCH=0
RESUME_MODEL=""
GPU_IDS="0,1,2,3"
SINGLE_GPU=false
BATCH_SIZE=2
AUTO_RESUME=true  # Automatically resume from latest checkpoint if available

# Help function
show_help() {
    cat << EOF
IM2ELEVATION Training Script

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (default: DFC2023Amini)
    -e, --epochs NUM            Number of epochs to train (default: 100)
    -lr, --learning-rate RATE   Learning rate (default: 0.0001)
    -o, --output DIR            Output directory for models (default: models_output)
    -c, --csv PATH              Path to training CSV file (auto-detected if not specified)
    -r, --resume EPOCH          Resume training from epoch (default: 0)
    -m, --model PATH            Path to model file for resuming (required if --resume > 0)
    --gpu-ids IDS               Comma-separated list of GPU IDs to use (default: 0,1,2,3)
    --single-gpu                Use single GPU for training
    -b, --batch-size NUM        Batch size per GPU for training (default: 2)
    --no-resume                 Start training from scratch (don't auto-resume from checkpoints)
    -h, --help                  Show this help message

Examples:
    # Basic training
    $0 --dataset DFC2023Amini --epochs 50

    # Use specific GPUs
    $0 --dataset DFC2023Amini --epochs 50 --gpu-ids "0,1"

    # Single GPU training
    $0 --dataset DFC2023Amini --epochs 50 --single-gpu

    # Resume training from epoch 30
    $0 --dataset DFC2023Amini --resume 30 --model models_output/DFC2023Amini/DFC2023Amini_model_29.pth.tar

    # Custom learning rate and output directory
    $0 --dataset contest --epochs 100 --learning-rate 0.0005 --output my_models
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--csv)
            CSV_PATH="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_EPOCH="$2"
            shift 2
            ;;
        -m|--model)
            RESUME_MODEL="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --single-gpu)
            SINGLE_GPU=true
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-resume)
            AUTO_RESUME=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Auto-detect CSV path if not provided
if [[ -z "$CSV_PATH" ]]; then
    CSV_PATH="./dataset/train_${DATASET_NAME}.csv"
fi

# Validate inputs
if [[ ! -f "$CSV_PATH" ]]; then
    echo "Error: Training CSV file not found: $CSV_PATH"
    echo "Available CSV files:"
    ls -la ./dataset/train_*.csv 2>/dev/null || echo "No training CSV files found in ./dataset/"
    exit 1
fi

if [[ $RESUME_EPOCH -gt 0 && -z "$RESUME_MODEL" ]]; then
    echo "Error: Model path required when resuming training (use --model option)"
    exit 1
fi

if [[ $RESUME_EPOCH -gt 0 && ! -f "$RESUME_MODEL" ]]; then
    echo "Error: Resume model file not found: $RESUME_MODEL"
    exit 1
fi

# Create dataset-specific output directory
DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
mkdir -p "$DATASET_OUTPUT_DIR"

# Auto-resume logic if enabled and no manual resume specified
if [[ "$AUTO_RESUME" == true && $RESUME_EPOCH -eq 0 && -z "$RESUME_MODEL" ]]; then
    # Look for latest checkpoint
    LATEST_CHECKPOINT=$(find "$DATASET_OUTPUT_DIR" -name "*_model_latest.pth.tar" 2>/dev/null | head -1)
    if [[ -n "$LATEST_CHECKPOINT" && -f "$LATEST_CHECKPOINT" ]]; then
        # Extract epoch number from checkpoint file
        LATEST_EPOCH=$(python -c "
import torch
try:
    checkpoint = torch.load('$LATEST_CHECKPOINT', map_location='cpu')
    print(checkpoint.get('epoch', 0) + 1)
except:
    print(0)
")
        if [[ $LATEST_EPOCH -gt 0 ]]; then
            echo "Found existing checkpoint: $(basename "$LATEST_CHECKPOINT")"
            echo "Auto-resuming training from epoch $LATEST_EPOCH"
            RESUME_EPOCH=$LATEST_EPOCH
            RESUME_MODEL="$LATEST_CHECKPOINT"
        fi
    fi
fi

# Print configuration
echo "======================================"
echo "IM2ELEVATION Training Configuration"
echo "======================================"
echo "Dataset:        $DATASET_NAME"
echo "Training CSV:   $CSV_PATH"
echo "Epochs:         $EPOCHS"
echo "Learning Rate:  $LEARNING_RATE"
echo "Output Dir:     $DATASET_OUTPUT_DIR"
echo "Batch Size:     $BATCH_SIZE"
if [[ "$SINGLE_GPU" == true ]]; then
    echo "GPU Mode:       Single GPU (GPU 0)"
else
    echo "GPU Mode:       Multi-GPU [$GPU_IDS]"
fi
echo "Resume Epoch:   $RESUME_EPOCH"
echo "Auto Resume:    $AUTO_RESUME"
if [[ $RESUME_EPOCH -gt 0 ]]; then
    echo "Resume Model:   $RESUME_MODEL"
fi
echo "======================================"

# Count training samples
TRAIN_SAMPLES=$(wc -l < "$CSV_PATH")
echo "Training samples: $TRAIN_SAMPLES"
echo ""

# Confirm before starting
read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Build training command
TRAIN_CMD="python train.py"
TRAIN_CMD="$TRAIN_CMD --data $DATASET_OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --csv $CSV_PATH"
TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
TRAIN_CMD="$TRAIN_CMD --lr $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --start_epoch $RESUME_EPOCH"

if [[ $RESUME_EPOCH -gt 0 ]]; then
    TRAIN_CMD="$TRAIN_CMD --model $RESUME_MODEL"
fi

# Add GPU options
if [[ "$SINGLE_GPU" == true ]]; then
    TRAIN_CMD="$TRAIN_CMD --single-gpu"
else
    TRAIN_CMD="$TRAIN_CMD --gpu-ids $GPU_IDS"
fi

# Start training
echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Execute training command with clean output
eval "$TRAIN_CMD"

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"
echo "Models saved in: $DATASET_OUTPUT_DIR"
echo "Check for 'best_epoch_*.pth.tar' and 'latest.pth.tar' files"
echo "======================================"
