#!/bin/bash

# IM2ELEVATION Testing Script
# Usage: ./test_model.sh [OPTIONS]

set -e  # Exit on any error

# Default values
DATASET_NAME="Dublin"
MODEL_BASE_DIR="pipeline_output"
MODEL_DIR=""
CSV_PATH=""
OUTPUT_FILE=""
SAVE_RESULTS=true
GPU_IDS="0"
SINGLE_GPU=false
BATCH_SIZE=1
DISABLE_NORMALIZATION=true  # Disable entire normalization pipeline

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
    --single-gpu                Use single GPU for testing
    --batch-size NUM            Batch size for testing (default: 1)
    --disable-normalization     Disable entire normalization pipeline (x1000, /100000, x100) for raw model analysis
    --no-save                   Don't save results to file (print to terminal only)
    -h, --help                  Show this help message

Examples:
    # Basic testing (uses pipeline_output/Dublin/)
    $0 --dataset Dublin

    # Test with custom base directory
    $0 --dataset Dublin --base-dir my_models

    # Test with specific model directory
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
        --single-gpu)
            SINGLE_GPU=true
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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
if [[ "$SINGLE_GPU" == true ]]; then
    echo "GPU Mode:       Single GPU (GPU 0)"
else
    echo "GPU Mode:       Multi-GPU [$GPU_IDS]"
fi
if [[ "$SAVE_RESULTS" == true ]]; then
    echo "Output File:    $OUTPUT_FILE"
else
    echo "Output:         Terminal only"
fi
echo "Disable Norm:   $DISABLE_NORMALIZATION"
echo "======================================"

# Count test samples
TEST_SAMPLES=$(wc -l < "$CSV_PATH")
echo "Test samples: $TEST_SAMPLES"
echo ""

# Show first few model files
echo "Model checkpoints found:"
for i in "${!MODEL_FILES[@]}"; do
    if [[ $i -lt 5 ]]; then
        echo "  $(basename "${MODEL_FILES[$i]}")"
    elif [[ $i -eq 5 ]]; then
        echo "  ... and $((${#MODEL_FILES[@]} - 5)) more"
        break
    fi
done
echo ""

# Confirm before starting
read -p "Start testing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing cancelled."
    exit 0
fi

# Build testing command
TEST_CMD="python test.py --model $MODEL_DIR --csv $CSV_PATH --batch-size $BATCH_SIZE"
if [[ "$DISABLE_NORMALIZATION" == true ]]; then
    TEST_CMD="$TEST_CMD --disable-normalization"
fi
if [[ "$SINGLE_GPU" == true ]]; then
    TEST_CMD="$TEST_CMD --single-gpu"
else
    TEST_CMD="$TEST_CMD --gpu-ids $GPU_IDS"
fi

# Start testing
echo "Starting testing..."
echo "Command: $TEST_CMD"
echo ""

if [[ "$SAVE_RESULTS" == true ]]; then
    echo "Results will be saved to: $OUTPUT_FILE"
    echo ""
    
    # Create header for results file
    {
        echo "IM2ELEVATION Test Results"
        echo "========================"
        echo "Dataset: $DATASET_NAME"
        echo "Test CSV: $CSV_PATH"
        echo "Model Dir: $MODEL_DIR"
        echo "Test Date: $(date)"
        echo "Test Samples: $TEST_SAMPLES"
        echo "Model Checkpoints: ${#MODEL_FILES[@]}"
        echo ""
        echo "Results:"
        echo "--------"
    } > "$OUTPUT_FILE"
    
    # Run testing with output to file and terminal
    eval "$TEST_CMD" 2>&1 | tee -a "$OUTPUT_FILE"
    
    echo ""
    echo "======================================"
    echo "Testing completed!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "======================================"
    
    # Show summary of results
    echo ""
    echo "Results Summary:"
    echo "==============="
    echo "Latest test results:"
    grep -E "Model Loss|MSE|RMSE|MAE|SSIM" "$OUTPUT_FILE" | tail -n 20
    
    # Show comparison with paper results based on dataset
    echo ""
    echo "Reference (from IM2ELEVATION paper):"
    case "${DATASET_NAME,,}" in
        dublin*)
            echo "Dublin dataset - MAE: 1.46m, RMSE: 3.05m"
            echo "Note: Significant differences may indicate need for longer training"
            ;;
        dfc2018*|*dfc2018*)
            echo "IEEE DFC2018 dataset - MAE: 1.19m, RMSE: 2.88m"
            ;;
        potsdam*|*potsdam*)
            echo "ISPRS Potsdam dataset - MAE: 1.52m, RMSE: 2.64m"
            ;;
        vaihingen*|*vaihingen*)
            echo "ISPRS Vaihingen dataset - RMSE: 4.66m (MAE not reported)"
            ;;
        *)
            echo "No specific benchmark available for dataset: $DATASET_NAME"
            echo "General note: Lower MAE and RMSE values indicate better performance"
            ;;
    esac
    
else
    # Run testing with terminal output only
    eval "$TEST_CMD"
    
    echo ""
    echo "======================================"
    echo "Testing completed!"
    echo "======================================"
fi
