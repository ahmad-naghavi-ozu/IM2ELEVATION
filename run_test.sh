#!/bin/bash

# IM2ELEVATION Testing Script
# Usage: ./test_model.sh [OPTIONS]

set -e  # Exit on any error

# Default values
DATASET_NAME="DFC2019_crp512_bin"
MODEL_BASE_DIR="pipeline_output"
MODEL_DIR=""
CSV_PATH=""
OUTPUT_FILE=""
SAVE_RESULTS=true
GPU_IDS="0"
BATCH_SIZE=1
ENABLE_CLIPPING=false
CLIPPING_THRESHOLD=30.0
DISABLE_TARGET_FILTERING=false
TARGET_THRESHOLD=1.0
DISABLE_NORMALIZATION=false  # Disable entire normalization pipeline - default false

# Help function
show_help() {
    cat << EOF
IM2ELEVATION Testing Script

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (default: Dublin)
    -b, --base-dir DIR          Base directory containing dataset subdirectories (default: pipeline_output)
    -m, --model-dir DIR         Specific model directory (overrides auto-detection)
    -c, --csv PATH              Path to test CSV file (auto-detected if not specified)
    -o, --output FILE           Output file for results (auto-generated if not specified)
    --gpu-ids IDS               Comma-separated list of GPU IDs to use (default: 0,1,2,3)
    --batch-size NUM            Batch size for testing (default: 1)
    --enable-clipping           Enable prediction clipping (disabled by default for full height range)
    --clipping-threshold NUM    Height threshold for clipping predictions in meters (default: 30.0)
    --disable-target-filtering  Disable target-based filtering of predictions (enabled by default)
    --target-threshold NUM      Target height threshold for filtering predictions in meters (default: 1.0)
    --disable-normalization     Disable entire normalization pipeline (x1000, /100000, x100) for raw model analysis
    --no-save                   Don't save results to file (print to terminal only)
    -h, --help                  Show this help message

Examples:
    # Basic testing with no clipping (default - allows full height range)
    $0 --dataset Dublin

    # Test with legacy clipping enabled (30m threshold)
    $0 --dataset Dublin --enable-clipping

    # Test with custom clipping threshold
    $0 --dataset Dublin --enable-clipping --clipping-threshold 50.0

    # Test with no target filtering (allows predictions on all areas)
    $0 --dataset Dublin --disable-target-filtering

    # Test with custom model directory
    $0 --dataset contest --model-dir my_models/contest_experiment1
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        -b|--base-dir)
            MODEL_BASE_DIR="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -c|--csv)
            CSV_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --enable-clipping)
            ENABLE_CLIPPING=true
            shift
            ;;
        --clipping-threshold)
            CLIPPING_THRESHOLD="$2"
            shift 2
            ;;
        --disable-target-filtering)
            DISABLE_TARGET_FILTERING=true
            shift
            ;;
        --target-threshold)
            TARGET_THRESHOLD="$2"
            shift 2
            ;;
        --disable-normalization)
            DISABLE_NORMALIZATION=true
            shift
            ;;
        --no-save)
            SAVE_RESULTS=false
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

# Auto-detect model directory if not provided
if [[ -z "$MODEL_DIR" ]]; then
    MODEL_DIR="${MODEL_BASE_DIR}/${DATASET_NAME}"
fi

# Validate required inputs
if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: Dataset name is required"
    show_help
    exit 1
fi

if [[ -z "$MODEL_DIR" ]]; then
    echo "Error: Model directory is required (use --model-dir option)"
    show_help
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Auto-detect CSV path if not provided
if [[ -z "$CSV_PATH" ]]; then
    CSV_PATH="./dataset/test_${DATASET_NAME}.csv"
fi

# Validate CSV file
if [[ ! -f "$CSV_PATH" ]]; then
    echo "Error: Test CSV file not found: $CSV_PATH"
    echo "Available CSV files:"
    ls -la ./dataset/test_*.csv 2>/dev/null || echo "No test CSV files found in ./dataset/"
    exit 1
fi

# Auto-generate output file if not provided
if [[ -z "$OUTPUT_FILE" && "$SAVE_RESULTS" == true ]]; then
    OUTPUT_FILE="${MODEL_DIR}/test_results_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).txt"
fi

# Check for model files
MODEL_FILES=($(find "$MODEL_DIR" -name "*.tar" | sort))
if [[ ${#MODEL_FILES[@]} -eq 0 ]]; then
    echo "Error: No model checkpoint files (.tar) found in $MODEL_DIR"
    exit 1
fi

# Print configuration
echo "======================================"
echo "IM2ELEVATION Testing Configuration"
echo "======================================"
echo "Dataset:        $DATASET_NAME"
echo "Model Dir:      $MODEL_DIR"
echo "Test CSV:       $CSV_PATH"
echo "Model Files:    ${#MODEL_FILES[@]} checkpoints found"
echo "Batch Size:     $BATCH_SIZE"
