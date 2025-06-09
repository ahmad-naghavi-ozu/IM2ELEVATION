#!/bin/bash

# IM2ELEVATION Testing Script
# Usage: ./test_model.sh [OPTIONS]

set# Auto-detect model directory if not provided
if [[ -z "$MODEL_DIR" ]]; then
    MODEL_DIR="${MODEL_BASE_DIR}/${DATASET_NAME}"
fi

# Validate inputs
if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: Dataset name is required"
    show_help
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    echo "Available dataset directories in $MODEL_BASE_DIR:"
    ls -la "$MODEL_BASE_DIR" 2>/dev/null || echo "Base directory $MODEL_BASE_DIR not found"
    exit 1
fi on any error

# Default values
DATASET_NAME="DFC2023Amini"
MODEL_BASE_DIR="models_output"
MODEL_DIR=""
CSV_PATH=""
OUTPUT_FILE=""
SAVE_RESULTS=true

# Help function
show_help() {
    cat << EOF
IM2ELEVATION Testing Script

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (default: DFC2023Amini)
    -b, --base-dir DIR          Base directory containing dataset subdirectories (default: models_output)
    -m, --model-dir DIR         Specific model directory (overrides auto-detection)
    -c, --csv PATH              Path to test CSV file (auto-detected if not specified)
    -o, --output FILE           Output file for results (auto-generated if not specified)
    --no-save                   Don't save results to file (print to terminal only)
    -h, --help                  Show this help message

Examples:
    # Basic testing (uses models_output/DFC2023Amini/)
    $0 --dataset DFC2023Amini

    # Test with custom base directory
    $0 --dataset DFC2023Amini --base-dir my_models

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

# Validate required inputs
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
    OUTPUT_FILE="test_results_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).txt"
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
if [[ "$SAVE_RESULTS" == true ]]; then
    echo "Output File:    $OUTPUT_FILE"
else
    echo "Output:         Terminal only"
fi
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
TEST_CMD="python test.py --model $MODEL_DIR --csv $CSV_PATH"

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
    grep -E "Model Loss|MSE|RMSE|MAE|SSIM" "$OUTPUT_FILE" | tail -n 20
    
else
    # Run testing with terminal output only
    eval "$TEST_CMD"
    
    echo ""
    echo "======================================"
    echo "Testing completed!"
    echo "======================================"
fi
