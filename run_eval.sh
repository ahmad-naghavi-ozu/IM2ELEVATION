#!/bin/bash

# run_eval.sh - Script to run complete evaluation pipeline for IM2ELEVATION
# Usage: ./run_eval.sh [OPTIONS]

set -e  # Exit on error

# Default values
DATASET_NAME="DFC2019_crp512_bin"
DATASET_PATH="/home/asfand/Ahmad/datasets/DFC2019_crp512_bin"
MODEL_PATH=""
GPU_IDS="1"
BATCH_SIZE=1
SKIP_PREDICTIONS=false
FORCE_REGENERATE=false
DISABLE_NORMALIZATION=true  # Disable entire normalization pipeline
USE_UINT16_CONVERSION=false  # Use uint16 conversion for depth data

# Clipping options
ENABLE_CLIPPING=false
CLIPPING_THRESHOLD=30.0
DISABLE_TARGET_FILTERING=false
TARGET_THRESHOLD=1.0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
IM2ELEVATION Evaluation Pipeline Script

This script runs the complete evaluation pipeline: Generate Predictions → Evaluate Metrics

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (required)
    -p, --dataset-path PATH     Full path to dataset directory (default: /home/asfand/Ahmad/datasets/DFC2019_crp512_bin)
    -m, --model-path PATH       Path to model directory (default: pipeline_output/DATASET_NAME)
    --gpu-ids IDS               Comma-separated list of GPU IDs to use (default: 0,1,2,3)
    -b, --batch-size NUM        Batch size for prediction generation (default: 3)
    --skip-predictions          Skip prediction generation (use existing predictions)
    --force-regenerate          Force regenerate predictions even if they exist
    --disable-normalization     Disable entire normalization pipeline (x1000, /100000, x100) for raw model analysis
    --uint16-conversion         Use uint16 conversion instead of float32 for depth data (original IM2ELEVATION format)
    -h, --help                  Show this help message

Available datasets:
    - DFC2023S
    - DFC2019_crp512_bin  
    - Dublin
    - Vaihingen
    - postdam

Examples:
    # Basic evaluation
    $0 --dataset DFC2023S

    # Use specific dataset path
    $0 --dataset DFC2023S --dataset-path /path/to/your/dataset

    # Custom model path and GPU settings
    $0 --dataset DFC2023S --model-path my_models/DFC2023S --gpu-ids 0,1

    # Skip prediction generation and use existing files
    $0 --dataset DFC2023S --skip-predictions

    # Force regenerate predictions
    $0 --dataset DFC2023S --force-regenerate
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        -p|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --skip-predictions)
            SKIP_PREDICTIONS=true
            shift
            ;;
        --force-regenerate)
            FORCE_REGENERATE=true
            shift
            ;;
        --disable-normalization)
            DISABLE_NORMALIZATION=true
            shift
            ;;
        --uint16-conversion)
            USE_UINT16_CONVERSION=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET_NAME" ]]; then
    print_error "Dataset name is required"
    show_help
    exit 1
fi

# Set default model path if not provided
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="pipeline_output/${DATASET_NAME}"
fi

# Validate dataset
VALID_DATASETS=("DFC2023S" "DFC2019_crp512_bin" "Dublin" "Vaihingen" "postdam")
if [[ ! " ${VALID_DATASETS[@]} " =~ " ${DATASET_NAME} " ]]; then
    print_error "Invalid dataset: $DATASET_NAME"
    print_error "Valid datasets: ${VALID_DATASETS[*]}"
    exit 1
fi

# Set paths
CSV_FILE="dataset/test_${DATASET_NAME}.csv"
PREDICTIONS_DIR="${MODEL_PATH}/predictions"

# Print evaluation configuration
print_status "=============================================="
print_status "IM2ELEVATION Evaluation Pipeline Configuration"
print_status "=============================================="
print_status "Dataset:        $DATASET_NAME"
print_status "Dataset Path:   $DATASET_PATH"
print_status "Model Path:     $MODEL_PATH"
print_status "CSV file:       $CSV_FILE"
print_status "Batch Size:     $BATCH_SIZE"
print_status "GPU Mode:       [$GPU_IDS]"
print_status ""
print_status "Pipeline Steps:"
print_status "  Generate Predictions: $([ "$SKIP_PREDICTIONS" == true ] && echo "SKIP" || echo "RUN")"
print_status "  Force Regenerate:     $([ "$FORCE_REGENERATE" == true ] && echo "YES" || echo "NO")"
print_status "  Disable Normalization: $([ "$DISABLE_NORMALIZATION" == true ] && echo "YES" || echo "NO")"
print_status "  Use Uint16 Conversion: $([ "$USE_UINT16_CONVERSION" == true ] && echo "YES" || echo "NO")"
print_status "  Evaluate Metrics:     RUN"
print_status "=============================================="
print_status ""

# Confirm before starting
read -p "Start evaluation pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled."
    exit 0
fi

# Check if required files exist
if [ ! -f "$CSV_FILE" ]; then
    print_error "CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model directory not found: $MODEL_PATH"
    exit 1
fi

# Check if model checkpoints exist
if ! ls "${MODEL_PATH}"/*.tar >/dev/null 2>&1; then
    print_error "No model checkpoints found in: $MODEL_PATH"
    exit 1
fi

print_status "All required files found, proceeding with evaluation..."

# Step 1: Generate predictions (if not already exist or if forced)
if [[ "$SKIP_PREDICTIONS" == false ]]; then
    if [[ "$FORCE_REGENERATE" == true ]] || [[ ! -d "$PREDICTIONS_DIR" ]] || [[ -z "$(ls -A $PREDICTIONS_DIR 2>/dev/null)" ]]; then
        if [[ "$FORCE_REGENERATE" == true ]] && [[ -d "$PREDICTIONS_DIR" ]]; then
            print_warning "Force regenerating predictions, removing existing directory..."
            rm -rf "$PREDICTIONS_DIR"
        fi
        
        print_status "Generating predictions..."
        
        # Build prediction command with GPU options
        PRED_CMD="python test.py --model \"$MODEL_PATH\" --csv \"$CSV_FILE\" --batch-size $BATCH_SIZE --save-predictions --gpu-ids $GPU_IDS"
        if [[ "$DISABLE_NORMALIZATION" == true ]]; then
            PRED_CMD="$PRED_CMD --disable-normalization"
        fi
        if [[ "$USE_UINT16_CONVERSION" == true ]]; then
            PRED_CMD="$PRED_CMD --uint16-conversion"
        fi
        
        # Note: GPU mode is automatically handled by test.py based on gpu-ids
        print_status "Command: $PRED_CMD"
        eval "$PRED_CMD"
        
        if [ $? -eq 0 ]; then
            print_success "Predictions generated successfully"
        else
            print_error "Failed to generate predictions"
            exit 1
        fi
    else
        print_warning "Predictions directory already exists and contains files"
        print_warning "Skipping prediction generation. Use --force-regenerate to overwrite"
    fi
else
    print_status "Skipping prediction generation as requested"
fi

# Step 2: Run evaluation
print_status "Running evaluation on saved predictions..."

# Build evaluation command with optional clipping parameters
EVAL_CMD="python evaluate.py --predictions-dir \"$PREDICTIONS_DIR\" --csv-file \"$CSV_FILE\" --dataset-name \"$DATASET_NAME\" --output-dir \"$MODEL_PATH\""

if [[ "$ENABLE_CLIPPING" == true ]]; then
    EVAL_CMD="$EVAL_CMD --enable-clipping --clipping-threshold $CLIPPING_THRESHOLD"
fi

if [[ "$DISABLE_TARGET_FILTERING" == true ]]; then
    EVAL_CMD="$EVAL_CMD --disable-target-filtering"
fi

# Always add target threshold (whether filtering is enabled or not)
EVAL_CMD="$EVAL_CMD --target-threshold $TARGET_THRESHOLD"

print_status "Command: $EVAL_CMD"
eval "$EVAL_CMD"

if [ $? -eq 0 ]; then
    print_success "Evaluation completed successfully"
    print_success "Results saved in: $MODEL_PATH"
else
    print_error "Evaluation failed"
    exit 1
fi

# Step 3: Summary
print_status "Evaluation pipeline completed!"
print_status "Dataset: $DATASET_NAME"
print_status "Predictions saved to: $PREDICTIONS_DIR"
print_status "Results saved to: $MODEL_PATH"

# Count prediction files
PRED_COUNT=$(find "$PREDICTIONS_DIR" -name "*_pred.npy" | wc -l)
print_status "Total prediction files: $PRED_COUNT"

# Show latest results file
LATEST_RESULTS=$(find "$MODEL_PATH" -name "evaluation_results_${DATASET_NAME}_*.txt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST_RESULTS" ]; then
    print_status "Latest results file: $LATEST_RESULTS"
    echo ""
    echo "=== QUICK SUMMARY ==="
    grep -E "(RMSE|MAE|Delta)" "$LATEST_RESULTS" | head -7
fi
