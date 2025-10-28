#!/bin/bash

# IM2ELEVATION Unified Testing & Evaluation Script
# Usage: ./run_test_unified.sh [OPTIONS]
#
# DEFAULT BEHAVIOR: Test + Evaluate Mode
#   - Runs model inference on test set
#   - Saves predictions to disk as .npy files
#   - Runs detailed evaluation metrics
#   - This is the RECOMMENDED mode for production use
#
# This replaces the old run_test.sh and run_eval.sh scripts

set -e  # Exit on error

# ============================================
# Default Configuration Values
# ============================================

# Dataset configuration
DATASET_NAME="DFC2019_crp512_bin"           # Name of the dataset to process
DATASET_PATH="/home/asfand/Ahmad/datasets/DFC2019_crp512_bin"  # Full path to dataset directory
MODEL_BASE_DIR="pipeline_output"            # Base directory where trained models are stored
MODEL_DIR=""                                 # Specific model directory (auto-detected if empty)
CSV_PATH=""                                  # Path to test CSV file (auto-detected if empty)

# Output configuration
OUTPUT_FILE=""                               # Output file for test results (auto-generated if empty)
SAVE_RESULTS=true                            # Save test results to file (vs. terminal only)

# GPU and performance settings
GPU_IDS="0"                                # Comma-separated list of GPU IDs to use
BATCH_SIZE=1                                 # Batch size for testing (lower = less memory)

# Operation mode flags (Test + Evaluate is DEFAULT for practical use)
SAVE_PREDICTIONS=true                        # Save prediction .npy files to disk for later evaluation (default: true - recommended)
EVALUATE_ONLY=false                          # Skip testing, only run evaluation on existing saved predictions
QUICK_MODE=false                             # Quick Test Mode: compute metrics in-memory without saving (for fast iteration)

# Clipping options - Control height prediction limits
ENABLE_CLIPPING=false                        # Clip predictions >= threshold to 0 (disabled = allow full height range)
CLIPPING_THRESHOLD=30.0                      # Height threshold in meters for clipping (default: 30.0m)

# Target filtering options - Mask predictions based on ground truth
DISABLE_TARGET_FILTERING=false               # Set to true to evaluate all areas (default: false = filter low targets)
TARGET_THRESHOLD=1.0                         # Ground truth threshold in meters - predictions where GT <= this are masked to 0

# Normalization options - Control data scaling behavior
DISABLE_NORMALIZATION=true                   # Disable x1000, /100000, x100 pipeline to see raw model outputs (default: true)
USE_UINT16_CONVERSION=false                  # Use original IM2ELEVATION format: depth = (depth*1000).astype(np.uint16)

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
IM2ELEVATION Unified Testing & Evaluation Script

This script runs in three modes:
  1. Test + Evaluate Mode (DEFAULT - recommended): Run inference, save predictions, and evaluate
  2. Evaluate Only Mode: Skip testing and evaluate existing saved predictions  
  3. Quick Test Mode (optional): Run inference without saving predictions (for fast iteration)

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (required)
    -p, --dataset-path PATH     Full path to dataset directory (default: /home/asfand/Ahmad/datasets/DFC2019_crp512_bin)
    -b, --base-dir DIR          Base directory containing dataset subdirectories (default: pipeline_output)
    -m, --model-dir DIR         Specific model directory (overrides auto-detection)
    -c, --csv PATH              Path to test CSV file (auto-detected if not specified)
    -o, --output FILE           Output file for results (auto-generated if not specified)
    
    Mode Options:
    --evaluate-only             Skip testing and only evaluate existing saved predictions (Evaluate Only Mode)
    --quick-mode                Quick test without saving predictions (for fast iteration/debugging)
    
    GPU & Performance:
    --gpu-ids IDS               Comma-separated list of GPU IDs to use (default: 0,1)
    --batch-size NUM            Batch size for testing (default: 1)
    
    Filtering Options:
    --enable-clipping           Enable prediction clipping (disabled by default for full height range)
    --clipping-threshold NUM    Height threshold for clipping predictions in meters (default: 30.0)
    --disable-target-filtering  Disable target-based filtering of predictions (enabled by default)
    --target-threshold NUM      Target height threshold for filtering predictions in meters (default: 1.0)
    
    Normalization Options:
    --disable-normalization     Disable entire normalization pipeline (x1000, /100000, x100) for raw model analysis
    --use-uint16-conversion     Use original IM2ELEVATION uint16 conversion: depth = (depth*1000).astype(np.uint16)
    
    Output Options:
    --no-save                   Don't save results to file (print to terminal only)
    
    -h, --help                  Show this help message

Available datasets:
    - DFC2023S
    - DFC2019_crp512_bin
    - Dublin
    - Vaihingen
    - postdam

Examples:
    # Test + Evaluate mode (DEFAULT - save predictions and run full evaluation)
    $0 --dataset DFC2023S

    # Quick test mode (compute in-memory, don't save - for fast iteration)
    $0 --dataset DFC2023S --quick-mode

    # Test with clipping enabled
    $0 --dataset DFC2023S --enable-clipping --clipping-threshold 50.0

    # Only evaluate existing saved predictions (skip testing/inference)
    $0 --dataset DFC2023S --evaluate-only

    # Custom model directory and GPU settings
    $0 --dataset DFC2023S --model-dir my_models/experiment1 --gpu-ids 0,1,2
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
        --save-predictions)
            SAVE_PREDICTIONS=true
            shift
            ;;
        --quick-mode)
            QUICK_MODE=true
            SAVE_PREDICTIONS=false
            shift
            ;;
        --evaluate-only)
            EVALUATE_ONLY=true
            shift
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
        --use-uint16-conversion)
            USE_UINT16_CONVERSION=true
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

# Auto-detect model directory if not provided
if [[ -z "$MODEL_DIR" ]]; then
    MODEL_DIR="${MODEL_BASE_DIR}/${DATASET_NAME}"
fi

# Validate model directory
if [[ ! -d "$MODEL_DIR" ]]; then
    print_error "Model directory not found: $MODEL_DIR"
    exit 1
fi

# Auto-detect CSV path if not provided
if [[ -z "$CSV_PATH" ]]; then
    CSV_PATH="./dataset/test_${DATASET_NAME}.csv"
fi

# Validate CSV file
if [[ ! -f "$CSV_PATH" ]]; then
    print_error "Test CSV file not found: $CSV_PATH"
    print_error "Available CSV files:"
    ls -la ./dataset/test_*.csv 2>/dev/null || print_error "No test CSV files found in ./dataset/"
    exit 1
fi

# Check for model files
MODEL_FILES=($(find "$MODEL_DIR" -name "*.tar" 2>/dev/null | sort))
if [[ ${#MODEL_FILES[@]} -eq 0 ]]; then
    print_error "No model checkpoint files (.tar) found in $MODEL_DIR"
    exit 1
fi

# Set paths
PREDICTIONS_DIR="${MODEL_DIR}/predictions"

# ============================================
# Determine Operation Mode
# ============================================
# Three modes available:
#   1. Test + Evaluate (DEFAULT) - Run inference, save predictions to disk, then evaluate with detailed metrics
#   2. Evaluate Only - Skip inference, evaluate existing saved predictions
#   3. Quick Test - Run inference and compute metrics in-memory (predictions not saved to disk - for fast iteration)

# Validate conflicting flags
if [[ "$QUICK_MODE" == true && "$EVALUATE_ONLY" == true ]]; then
    print_error "Error: --quick-mode and --evaluate-only are mutually exclusive"
    exit 1
fi

if [[ "$EVALUATE_ONLY" == true ]]; then
    MODE="Evaluate Only"
    RUN_TESTING=false
    RUN_EVALUATION=true
elif [[ "$QUICK_MODE" == true ]]; then
    MODE="Quick Test"
    RUN_TESTING=true
    RUN_EVALUATION=false
    SAVE_PREDICTIONS=false
else
    MODE="Test + Evaluate"
    RUN_TESTING=true
    RUN_EVALUATION=true
    SAVE_PREDICTIONS=true
fi

# Auto-generate output file if not provided
if [[ -z "$OUTPUT_FILE" && "$SAVE_RESULTS" == true ]]; then
    OUTPUT_FILE="${MODEL_DIR}/test_results_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).txt"
fi

# Validate datasets
VALID_DATASETS=("DFC2023S" "DFC2019_crp512_bin" "Dublin" "Vaihingen" "postdam")
if [[ ! " ${VALID_DATASETS[@]} " =~ " ${DATASET_NAME} " ]]; then
    print_warning "Dataset '${DATASET_NAME}' not in known list: ${VALID_DATASETS[*]}"
    print_warning "Proceeding anyway..."
fi

# Print configuration
print_status "=============================================="
print_status "IM2ELEVATION Unified Testing Configuration"
print_status "=============================================="
print_status "Operation Mode: $MODE"
print_status "Dataset:        $DATASET_NAME"
print_status "Dataset Path:   $DATASET_PATH"
print_status "Model Dir:      $MODEL_DIR"
print_status "Test CSV:       $CSV_PATH"
print_status "Model Files:    ${#MODEL_FILES[@]} checkpoints found"
print_status "Batch Size:     $BATCH_SIZE"
print_status "GPU Mode:       [$GPU_IDS]"
print_status ""
print_status "Configuration:"
print_status "  Disable Normalization: $DISABLE_NORMALIZATION"
print_status "  Use Uint16 Conversion: $USE_UINT16_CONVERSION"
print_status "  Enable Clipping:       $ENABLE_CLIPPING"
if [[ "$ENABLE_CLIPPING" == true ]]; then
    print_status "  Clipping Threshold:    $CLIPPING_THRESHOLD"
fi
print_status "  Target Filtering:      $([ "$DISABLE_TARGET_FILTERING" == true ] && echo "DISABLED" || echo "ENABLED")"
print_status "  Target Threshold:      $TARGET_THRESHOLD"
print_status ""
print_status "Pipeline Steps:"
print_status "  Run Testing:            $([ "$RUN_TESTING" == true ] && echo "YES" || echo "NO")"
if [[ "$RUN_TESTING" == true ]]; then
    print_status "    - Save Predictions:   $SAVE_PREDICTIONS"
fi
print_status "  Run Evaluation:         $([ "$RUN_EVALUATION" == true ] && echo "YES" || echo "NO")"
if [[ "$SAVE_RESULTS" == true && "$RUN_TESTING" == true ]]; then
    print_status "  Output File:            $OUTPUT_FILE"
fi
print_status "=============================================="
print_status ""

# Confirm before starting
read -p "Start pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# ============================================
# STEP 1: RUN TESTING (if needed)
# ============================================
# This step runs model inference on the test set and computes predictions
# In Quick Test Mode: predictions computed in-memory only (not saved to disk)
# In Test + Evaluate Mode: predictions saved as .npy files for later evaluation
if [[ "$RUN_TESTING" == true ]]; then
    print_status "=========================================="
    print_status "Step 1: Running Model Testing"
    print_status "=========================================="
    
    # Build testing command
    TEST_CMD="python test.py --model \"$MODEL_DIR\" --csv \"$CSV_PATH\" --batch-size $BATCH_SIZE --gpu-ids $GPU_IDS"
    
    # Add save predictions flag if needed
    if [[ "$SAVE_PREDICTIONS" == true ]]; then
        # Always regenerate - delete existing predictions if they exist
        if [[ -d "$PREDICTIONS_DIR" ]] && [[ -n "$(ls -A $PREDICTIONS_DIR 2>/dev/null)" ]]; then
            print_warning "Predictions directory exists. Removing and regenerating..."
            rm -rf "$PREDICTIONS_DIR"
        fi
        
        TEST_CMD="$TEST_CMD --save-predictions"
        print_status "Predictions will be saved to: $PREDICTIONS_DIR"
    fi
    
    # Add normalization flags (controls data scaling behavior)
    if [[ "$DISABLE_NORMALIZATION" == true ]]; then
        TEST_CMD="$TEST_CMD --disable-normalization"
    fi
    
    if [[ "$USE_UINT16_CONVERSION" == true ]]; then
        TEST_CMD="$TEST_CMD --uint16-conversion"
    fi
    
    # Add clipping options (control height prediction limits)
    if [[ "$ENABLE_CLIPPING" == true ]]; then
        TEST_CMD="$TEST_CMD --enable-clipping --clipping-threshold $CLIPPING_THRESHOLD"
    fi
    
    # Add target filtering options (mask predictions based on ground truth)
    if [[ "$DISABLE_TARGET_FILTERING" == true ]]; then
        TEST_CMD="$TEST_CMD --disable-target-filtering"
    fi
    
    TEST_CMD="$TEST_CMD --target-threshold $TARGET_THRESHOLD"
    
    print_status "Command: $TEST_CMD"
    print_status ""
    
    # Execute test command
    if [[ "$SAVE_RESULTS" == true ]]; then
        # Create results file header
        {
            echo "IM2ELEVATION Test Results"
            echo "========================"
            echo "Mode: $MODE"
            echo "Dataset: $DATASET_NAME"
            echo "Model Dir: $MODEL_DIR"
            echo "Test CSV: $CSV_PATH"
            echo "Test Date: $(date)"
            echo "Model Checkpoints: ${#MODEL_FILES[@]}"
            echo "Batch Size: $BATCH_SIZE"
            echo "GPU IDs: $GPU_IDS"
            echo "Disable Normalization: $DISABLE_NORMALIZATION"
            echo "Use Uint16 Conversion: $USE_UINT16_CONVERSION"
            echo "Save Predictions: $SAVE_PREDICTIONS"
            echo ""
            echo "Results:"
            echo "--------"
        } > "$OUTPUT_FILE"
        
        eval "$TEST_CMD" 2>&1 | tee -a "$OUTPUT_FILE"
        
        if [ $? -eq 0 ]; then
            print_success "Testing completed successfully"
            if [[ "$SAVE_PREDICTIONS" == true ]]; then
                PRED_COUNT=$(find "$PREDICTIONS_DIR" -name "*_pred.npy" 2>/dev/null | wc -l)
                print_success "Generated $PRED_COUNT prediction files"
            fi
        else
            print_error "Testing failed"
            exit 1
        fi
    else
        eval "$TEST_CMD"
        
        if [ $? -eq 0 ]; then
            print_success "Testing completed successfully"
        else
            print_error "Testing failed"
            exit 1
        fi
    fi
    
    print_status ""
else
    print_status "=========================================="
    print_status "Step 1: Skipping Testing (Evaluate Only Mode)"
    print_status "=========================================="
    print_status ""
fi

# ============================================
# STEP 2: RUN EVALUATION (if needed)
# ============================================
# This step loads saved predictions and calculates detailed metrics
# Requires prediction .npy files generated by test.py with --save-predictions
if [[ "$RUN_EVALUATION" == true ]]; then
    print_status "=========================================="
    print_status "Step 2: Running Evaluation Metrics"
    print_status "=========================================="
    
    # Check if predictions directory exists
    if [[ ! -d "$PREDICTIONS_DIR" ]]; then
        print_error "Predictions directory not found: $PREDICTIONS_DIR"
        print_error "Cannot run evaluation without predictions"
        exit 1
    fi
    
    # Check if predictions exist
    PRED_COUNT=$(find "$PREDICTIONS_DIR" -name "*_pred.npy" 2>/dev/null | wc -l)
    if [[ $PRED_COUNT -eq 0 ]]; then
        print_error "No prediction files found in: $PREDICTIONS_DIR"
        print_error "Cannot run evaluation without predictions"
        exit 1
    fi
    
    print_status "Found $PRED_COUNT prediction files"
    print_status "Running evaluation on saved predictions..."
    
    # Build evaluation command
    EVAL_CMD="python evaluate.py --predictions-dir \"$PREDICTIONS_DIR\" --csv-file \"$CSV_PATH\" --dataset-name \"$DATASET_NAME\" --output-dir \"$MODEL_DIR\""
    
    # Add clipping options (must match settings used during prediction generation)
    if [[ "$ENABLE_CLIPPING" == true ]]; then
        EVAL_CMD="$EVAL_CMD --enable-clipping --clipping-threshold $CLIPPING_THRESHOLD"
    fi
    
    # Add target filtering options (must match settings used during prediction generation)
    if [[ "$DISABLE_TARGET_FILTERING" == true ]]; then
        EVAL_CMD="$EVAL_CMD --disable-target-filtering"
    fi
    
    EVAL_CMD="$EVAL_CMD --target-threshold $TARGET_THRESHOLD"
    
    print_status "Command: $EVAL_CMD"
    print_status ""
    
    eval "$EVAL_CMD"
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation completed successfully"
        print_success "Results saved in: $MODEL_DIR"
        
        # Show latest evaluation results
        LATEST_EVAL_RESULTS=$(find "$MODEL_DIR" -name "evaluation_results_${DATASET_NAME}_*.txt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
        if [[ -n "$LATEST_EVAL_RESULTS" ]]; then
            print_status ""
            print_status "=== EVALUATION SUMMARY ==="
            grep -E "(RMSE|MAE|δ[₁₂₃])" "$LATEST_EVAL_RESULTS" 2>/dev/null | head -7 || echo "Evaluation results saved to file"
            print_status ""
            print_status "Full results: $LATEST_EVAL_RESULTS"
        fi
    else
        print_error "Evaluation failed"
        exit 1
    fi
    
    print_status ""
else
    print_status "=========================================="
    print_status "Step 2: Skipping Evaluation"
    print_status "=========================================="
    print_status ""
fi

# ============================================
# FINAL SUMMARY
# ============================================
# Display summary of completed operations and output locations
print_status "=============================================="
print_success "Pipeline completed successfully!"
print_status "=============================================="
print_status "Mode:           $MODE"
print_status "Dataset:        $DATASET_NAME"
print_status "Model Dir:      $MODEL_DIR"
if [[ "$RUN_TESTING" == true && "$SAVE_RESULTS" == true ]]; then
    print_status "Test Results:   $OUTPUT_FILE"
fi
if [[ "$SAVE_PREDICTIONS" == true || "$EVALUATE_ONLY" == true ]]; then
    print_status "Predictions:    $PREDICTIONS_DIR ($PRED_COUNT files)"
fi
if [[ "$RUN_EVALUATION" == true ]]; then
    print_status "Evaluation:     Check $MODEL_DIR for evaluation_results_*.txt"
fi
print_status "=============================================="
