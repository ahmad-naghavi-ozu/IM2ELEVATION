#!/bin/bash

# IM2ELEVATION Full Pipeline Script
# Usage: ./run_pipeline.sh [OPTIONS]

set -e  # Exit on any error

# Default values
DATASET_NAME="DFC2023Amini"
DATASET_PATH=""
EPOCHS=100
LEARNING_RATE=0.0001
OUTPUT_DIR="pipeline_output"
SKIP_CSV_GENERATION=false
SKIP_TRAINING=false
SKIP_TESTING=false

# Help function
show_help() {
    cat << EOF
IM2ELEVATION Full Pipeline Script

This script runs the complete pipeline: CSV generation → Training → Testing

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (default: DFC2023Amini)
    -p, --dataset-path PATH     Path to dataset root directory (required unless --skip-csv)
    -e, --epochs NUM            Number of epochs to train (default: 100)
    -lr, --learning-rate RATE   Learning rate (default: 0.0001)
    -o, --output DIR            Output directory for models and results (default: pipeline_output)
    --skip-csv                  Skip CSV generation (use existing CSV files)
    --skip-training             Skip training (use existing models for testing)
    --skip-testing              Skip testing (only generate CSV and train)
    -h, --help                  Show this help message

Examples:
    # Full pipeline
    $0 --dataset DFC2023Amini --dataset-path /path/to/DFC2023Amini --epochs 50

    # Skip CSV generation and use existing files
    $0 --dataset contest --skip-csv --epochs 100

    # Only generate CSV files
    $0 --dataset mydata --dataset-path /path/to/mydata --skip-training --skip-testing

    # Only test existing models
    $0 --dataset DFC2023Amini --skip-csv --skip-training
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
        --skip-csv)
            SKIP_CSV_GENERATION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-testing)
            SKIP_TESTING=true
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

# Validate inputs
if [[ "$SKIP_CSV_GENERATION" == false && -z "$DATASET_PATH" ]]; then
    echo "Error: Dataset path is required unless --skip-csv is used"
    show_help
    exit 1
fi

if [[ "$SKIP_CSV_GENERATION" == false && ! -d "$DATASET_PATH" ]]; then
    echo "Error: Dataset path not found: $DATASET_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print pipeline configuration
echo "=============================================="
echo "IM2ELEVATION Full Pipeline Configuration"
echo "=============================================="
echo "Dataset:        $DATASET_NAME"
if [[ "$SKIP_CSV_GENERATION" == false ]]; then
    echo "Dataset Path:   $DATASET_PATH"
fi
echo "Output Dir:     $OUTPUT_DIR"
echo "Epochs:         $EPOCHS"
echo "Learning Rate:  $LEARNING_RATE"
echo ""
echo "Pipeline Steps:"
echo "  CSV Generation: $([ "$SKIP_CSV_GENERATION" == true ] && echo "SKIP" || echo "RUN")"
echo "  Training:       $([ "$SKIP_TRAINING" == true ] && echo "SKIP" || echo "RUN")"
echo "  Testing:        $([ "$SKIP_TESTING" == true ] && echo "SKIP" || echo "RUN")"
echo "=============================================="
echo ""

# Confirm before starting
read -p "Start pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# Create pipeline log
PIPELINE_LOG="${OUTPUT_DIR}/pipeline_log_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "IM2ELEVATION Pipeline Log"
    echo "========================"
    echo "Dataset: $DATASET_NAME"
    echo "Start Time: $(date)"
    echo "Configuration:"
    echo "  Dataset Path: $DATASET_PATH"
    echo "  Output Dir: $OUTPUT_DIR"
    echo "  Epochs: $EPOCHS"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Skip CSV: $SKIP_CSV_GENERATION"
    echo "  Skip Training: $SKIP_TRAINING"
    echo "  Skip Testing: $SKIP_TESTING"
    echo ""
} > "$PIPELINE_LOG"

echo "Pipeline log: $PIPELINE_LOG"
echo ""

# Step 1: CSV Generation
if [[ "$SKIP_CSV_GENERATION" == false ]]; then
    echo "=========================================="
    echo "Step 1: Generating CSV files"
    echo "=========================================="
    
    CSV_CMD="python generate_dataset_csv.py \"$DATASET_PATH\" --dataset-name $DATASET_NAME"
    echo "Command: $CSV_CMD"
    echo ""
    
    {
        echo "Step 1: CSV Generation"
        echo "======================"
        echo "Command: $CSV_CMD"
        echo "Start: $(date)"
        echo ""
    } >> "$PIPELINE_LOG"
    
    eval "$CSV_CMD" 2>&1 | tee -a "$PIPELINE_LOG"
    
    {
        echo ""
        echo "CSV Generation completed: $(date)"
        echo ""
    } >> "$PIPELINE_LOG"
    
    echo "CSV generation completed!"
    echo ""
else
    echo "=========================================="
    echo "Step 1: Skipping CSV generation"
    echo "=========================================="
    echo "Using existing CSV files"
    echo ""
    
    {
        echo "Step 1: CSV Generation - SKIPPED"
        echo "================================"
        echo "Using existing CSV files"
        echo ""
    } >> "$PIPELINE_LOG"
fi

# Step 2: Training
if [[ "$SKIP_TRAINING" == false ]]; then
    echo "=========================================="
    echo "Step 2: Training model"
    echo "=========================================="
    
    TRAIN_CSV="./dataset/train_${DATASET_NAME}.csv"
    if [[ ! -f "$TRAIN_CSV" ]]; then
        echo "Error: Training CSV not found: $TRAIN_CSV"
        exit 1
    fi
    
    # Create dataset-specific output directory
    DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
    mkdir -p "$DATASET_OUTPUT_DIR"
    
    TRAIN_CMD="python train.py --data $DATASET_OUTPUT_DIR --csv $TRAIN_CSV --epochs $EPOCHS --lr $LEARNING_RATE"
    echo "Command: $TRAIN_CMD"
    echo ""
    
    {
        echo "Step 2: Training"
        echo "==============="
        echo "Command: $TRAIN_CMD"
        echo "Start: $(date)"
        echo ""
    } >> "$PIPELINE_LOG"
    
    # Execute training with clean output
    eval "$TRAIN_CMD" 2>&1 | tee -a "$PIPELINE_LOG"
    
    # Check if training actually succeeded by looking for model files
    DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
    TRAINED_MODEL_FILES=($(find "$DATASET_OUTPUT_DIR" -name "*.tar" 2>/dev/null | sort))
    
    if [[ ${#TRAINED_MODEL_FILES[@]} -eq 0 ]]; then
        echo "ERROR: Training failed - no model files were generated!"
        echo "Check the log file for details: $PIPELINE_LOG"
        exit 1
    fi
    
    {
        echo ""
        echo "Training completed: $(date)"
        echo "Generated ${#TRAINED_MODEL_FILES[@]} model checkpoints"
        echo ""
    } >> "$PIPELINE_LOG"
    
    echo "Training completed!"
    echo ""
else
    echo "=========================================="
    echo "Step 2: Skipping training"
    echo "=========================================="
    echo "Using existing model files"
    echo ""
    
    {
        echo "Step 2: Training - SKIPPED"
        echo "========================="
        echo "Using existing model files"
        echo ""
    } >> "$PIPELINE_LOG"
fi

# Step 3: Testing
if [[ "$SKIP_TESTING" == false ]]; then
    echo "=========================================="
    echo "Step 3: Testing model"
    echo "=========================================="
    
    TEST_CSV="./dataset/test_${DATASET_NAME}.csv"
    if [[ ! -f "$TEST_CSV" ]]; then
        echo "Error: Test CSV not found: $TEST_CSV"
        exit 1
    fi
    
    # Use dataset-specific output directory for testing
    DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
    
    # Check for model files
    MODEL_FILES=($(find "$DATASET_OUTPUT_DIR" -name "*.tar" 2>/dev/null | sort))
    if [[ ${#MODEL_FILES[@]} -eq 0 ]]; then
        echo "Error: No model checkpoint files found in $DATASET_OUTPUT_DIR"
        exit 1
    fi
    
    TEST_CMD="python test.py --model $DATASET_OUTPUT_DIR --csv $TEST_CSV"
    echo "Command: $TEST_CMD"
    echo "Found ${#MODEL_FILES[@]} model checkpoints to test"
    echo ""
    
    RESULTS_FILE="${DATASET_OUTPUT_DIR}/test_results_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Step 3: Testing"
        echo "==============="
        echo "Command: $TEST_CMD"
        echo "Model checkpoints: ${#MODEL_FILES[@]}"
        echo "Results file: $RESULTS_FILE"
        echo "Start: $(date)"
        echo ""
    } >> "$PIPELINE_LOG"
    
    # Create results file header
    {
        echo "IM2ELEVATION Test Results"
        echo "========================"
        echo "Dataset: $DATASET_NAME"
        echo "Model Dir: $DATASET_OUTPUT_DIR"
        echo "Test CSV: $TEST_CSV"
        echo "Test Date: $(date)"
        echo "Model Checkpoints: ${#MODEL_FILES[@]}"
        echo ""
        echo "Results:"
        echo "--------"
    } > "$RESULTS_FILE"
    
    eval "$TEST_CMD" 2>&1 | tee -a "$RESULTS_FILE" | tee -a "$PIPELINE_LOG"
    
    {
        echo ""
        echo "Testing completed: $(date)"
        echo "Results saved to: $RESULTS_FILE"
        echo ""
    } >> "$PIPELINE_LOG"
    
    echo "Testing completed!"
    echo "Results saved to: $RESULTS_FILE"
    echo ""
else
    echo "=========================================="
    echo "Step 3: Skipping testing"
    echo "=========================================="
    echo ""
    
    {
        echo "Step 3: Testing - SKIPPED"
        echo "========================"
        echo ""
    } >> "$PIPELINE_LOG"
fi

# Pipeline completion
{
    echo "Pipeline completed: $(date)"
    echo "================================="
} >> "$PIPELINE_LOG"

echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Pipeline log: $PIPELINE_LOG"
if [[ "$SKIP_TESTING" == false ]]; then
    echo ""
    echo "Test Results Summary:"
    echo "===================="
    # Show only the most recent test results (not all historical results)
    LATEST_RESULTS=$(find "$OUTPUT_DIR" -name "test_results_*.txt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [[ -n "$LATEST_RESULTS" ]]; then
        echo "Latest results from: $(basename "$LATEST_RESULTS")"
        # Show only the actual metrics (last line with Loss, MSE, RMSE, etc.)
        grep "Model Loss" "$LATEST_RESULTS" | tail -1 2>/dev/null || echo "Results saved to file"
    else
        echo "No test results found"
    fi
fi
echo "=============================================="
