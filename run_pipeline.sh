#!/bin/bash

# IM2ELEVATION Full Pipeline Script
# Usage: ./run_pipeline.sh [OPTIONS]

set -e  # Exit on any error

# Default values
DATASET_NAME="DFC2019_crp512_bin"
DATASET_PATH="/home/asfand/Ahmad/datasets/DFC2019_crp512_bin"
EPOCHS=100
LEARNING_RATE=0.0001
OUTPUT_DIR="pipeline_output"
SKIP_CSV_GENERATION=false
SKIP_TRAINING=false
SKIP_TESTING=false
SKIP_EVALUATION=false
GPU_IDS="0,1"
BATCH_SIZE=2 # Reduced default batch size to prevent OOM errors
AUTO_RESUME=true  # Automatically resume from latest checkpoint if available
FORCE_REGENERATE_PREDICTIONS=true  # Force regenerate predictions during evaluation
DISABLE_NORMALIZATION=true  # Disable entire normalization pipeline (x1000, /100000, x100)
USE_UINT16_CONVERSION=false  # Use original IM2ELEVATION uint16 conversion: depth = (depth*1000).astype(np.uint16)

# Clipping options for testing and evaluation
ENABLE_CLIPPING=false
CLIPPING_THRESHOLD=30.0
DISABLE_TARGET_FILTERING=false
TARGET_THRESHOLD=1.0

# Help function
show_help() {
    cat << EOF
IM2ELEVATION Full Pipeline Script

This script runs the complete pipeline: CSV generation → Training → Testing → Evaluation

Usage: $0 [OPTIONS]

Options:
    -d, --dataset NAME          Dataset name (default: Dublin)
    -p, --dataset-path PATH     Path to dataset root directory (required unless --skip-csv)
    -e, --epochs NUM            Number of epochs to train (default: 50)
    -lr, --learning-rate RATE   Learning rate (default: 0.0001)
    -o, --output DIR            Output directory for models and results (default: pipeline_output)
    --skip-csv                  Skip CSV generation (use existing CSV files)
    --skip-training             Skip training (use existing models for testing)
    --skip-testing              Skip testing (only generate CSV and train)
    --skip-evaluation           Skip evaluation (only generate CSV, train, and test)
    --gpu-ids IDS               Comma-separated list of GPU IDs to use (default: 0,1)
    --no-resume                 Start training from scratch (don't auto-resume from checkpoints)
    --force-regenerate          Force regenerate predictions during evaluation
    --disable-normalization     Disable entire normalization pipeline (x1000, /100000, x100) for raw model analysis
    --use-uint16-conversion     Use original IM2ELEVATION uint16 conversion: depth = (depth*1000).astype(np.uint16)
    -b, --batch-size NUM        Batch size per GPU for training, total batch size for testing (default: 1)
    
    Clipping Options (for testing and evaluation):
    --enable-clipping           Enable clipping of predictions >= threshold (default: disabled)
    --clipping-threshold NUM    Threshold for clipping predictions (default: 30.0)
    --disable-target-filtering  Disable filtering targets <= threshold (default: enabled)
    --target-threshold NUM      Threshold for target filtering (default: 1.0)
    
    -h, --help                  Show this help message

Examples:
    # Full pipeline with evaluation
    $0 --dataset DFC2023Amini --dataset-path /path/to/DFC2023Amini --epochs 50

    # Skip CSV generation and use existing files
    $0 --dataset contest --skip-csv --epochs 100

    # Only generate CSV files
    $0 --dataset mydata --dataset-path /path/to/mydata --skip-training --skip-testing --skip-evaluation

    # Only test and evaluate existing models
    $0 --dataset DFC2023Amini --skip-csv --skip-training

    # Use specific GPU (single GPU automatically detected)
    $0 --dataset DFC2023S --gpu-ids 2

    # Use multiple GPUs (multi-GPU automatically detected)
    $0 --dataset DFC2023S --gpu-ids 0,1,2

    # Force regenerate predictions during evaluation
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
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --no-resume)
            AUTO_RESUME=false
            shift
            ;;
        --force-regenerate)
            FORCE_REGENERATE_PREDICTIONS=true
            shift
            ;;
        --disable-normalization)
            DISABLE_NORMALIZATION=true
            shift
            ;;
        --use-uint16-conversion)
            USE_UINT16_CONVERSION=true
            shift
            ;;
        -b|--batch-size)
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

# Create dataset-specific output directory early for log placement
DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
mkdir -p "$DATASET_OUTPUT_DIR"

# Create pipeline log in dataset-specific directory
PIPELINE_LOG="${DATASET_OUTPUT_DIR}/pipeline_log_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).txt"

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
echo "Batch Size:     $BATCH_SIZE"

# Determine GPU mode based on the number of GPUs specified
GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [[ $GPU_COUNT -eq 1 ]]; then
    echo "GPU Mode:       Single GPU ($GPU_IDS)"
else
    echo "GPU Mode:       Multi-GPU [$GPU_IDS]"
fi

echo "Auto Resume:    $AUTO_RESUME"
echo "Disable Norm:   $DISABLE_NORMALIZATION"
echo "Uint16 Conv:    $USE_UINT16_CONVERSION"
echo ""

# Check GPU memory status before starting (if utility available)
if command -v python >/dev/null 2>&1 && [ -f "gpu_memory_manager.py" ]; then
    echo "GPU Memory Status Before Pipeline:"
    python gpu_memory_manager.py --info --suggest-batch-size 2>/dev/null || {
        echo "GPU memory manager not available, continuing without memory check..."
        python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "CUDA not available"
    }
    echo ""
else
    echo "GPU memory manager not available, skipping memory check..."
    echo ""
fi

echo "Pipeline Steps:"
echo "  CSV Generation: $([ "$SKIP_CSV_GENERATION" == true ] && echo "SKIP" || echo "RUN")"
echo "  Training:       $([ "$SKIP_TRAINING" == true ] && echo "SKIP" || echo "RUN")"
echo "  Testing:        $([ "$SKIP_TESTING" == true ] && echo "SKIP" || echo "RUN")"
echo "  Evaluation:     $([ "$SKIP_EVALUATION" == true ] && echo "SKIP" || echo "RUN")"
echo "=============================================="
echo ""

# Confirm before starting
read -p "Start pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# Pipeline log was already created above
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
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Skip CSV: $SKIP_CSV_GENERATION"
    echo "  Skip Training: $SKIP_TRAINING"
    echo "  Skip Testing: $SKIP_TESTING"
    echo "  Skip Evaluation: $SKIP_EVALUATION"
    echo "  GPU IDs: $GPU_IDS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Force Regenerate Predictions: $FORCE_REGENERATE_PREDICTIONS"
    echo "  Disable Normalization: $DISABLE_NORMALIZATION"
    echo "  Use Uint16 Conversion: $USE_UINT16_CONVERSION"
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
    
    # Dataset-specific output directory was already created earlier
    # Just reference it here
    
    # Check for existing checkpoints and auto-resume if enabled
    RESUME_ARGS="--start_epoch 0"  # Default to epoch 0
    if [[ "$AUTO_RESUME" == true ]]; then
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
                echo "Last completed epoch: $((LATEST_EPOCH - 1))"
                if [[ $LATEST_EPOCH -ge $EPOCHS ]]; then
                    echo "Training already completed ($((LATEST_EPOCH - 1)) epochs >= $EPOCHS target epochs)"
                    echo "Skipping training. Use --no-resume to start from scratch."
                    SKIP_TRAINING=true
                else
                    echo "Resuming training from epoch $LATEST_EPOCH to epoch $EPOCHS"
                    RESUME_ARGS="--start_epoch $LATEST_EPOCH --model $LATEST_CHECKPOINT"
                fi
            fi
        fi
    else
        # When auto-resume is disabled, always start from epoch 0
        echo "Auto-resume disabled, starting training from scratch"
        RESUME_ARGS="--start_epoch 0"
    fi
    
    # Build training command with GPU options and memory management
    TRAIN_CMD="python train.py --data $DATASET_OUTPUT_DIR --csv $TRAIN_CSV --epochs $EPOCHS --lr $LEARNING_RATE --batch-size $BATCH_SIZE --gpu-ids $GPU_IDS $RESUME_ARGS"
    
    # Add normalization flag if enabled
    if [[ "$DISABLE_NORMALIZATION" == true ]]; then
        TRAIN_CMD="$TRAIN_CMD --disable-normalization"
    fi
    
    # Add uint16 conversion flag if enabled
    if [[ "$USE_UINT16_CONVERSION" == true ]]; then
        TRAIN_CMD="$TRAIN_CMD --uint16-conversion"
    fi
    
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
    
    # Clear GPU memory after training
    echo "Clearing GPU memory after training..."
    if [ -f "gpu_memory_manager.py" ]; then
        python gpu_memory_manager.py --clear
    else
        python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || echo "Could not clear GPU cache"
    fi
    
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
    
    # Clear GPU memory before testing
    echo "Clearing GPU memory cache before testing..."
    if [ -f "gpu_memory_manager.py" ]; then
        python gpu_memory_manager.py --clear
    else
        python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || echo "Could not clear GPU cache"
    fi
    
    # Build testing command with GPU options and memory management
    TEST_CMD="python test.py --model $DATASET_OUTPUT_DIR --csv $TEST_CSV --batch-size $BATCH_SIZE --gpu-ids $GPU_IDS"
    if [[ "$DISABLE_NORMALIZATION" == true ]]; then
        TEST_CMD="$TEST_CMD --disable-normalization"
    fi
    
    # Add uint16 conversion flag if enabled
    if [[ "$USE_UINT16_CONVERSION" == true ]]; then
        TEST_CMD="$TEST_CMD --uint16-conversion"
    fi
    
    # Add clipping options
    if [[ "$ENABLE_CLIPPING" == true ]]; then
        TEST_CMD="$TEST_CMD --enable-clipping --clipping-threshold $CLIPPING_THRESHOLD"
    fi
    if [[ "$DISABLE_TARGET_FILTERING" == true ]]; then
        TEST_CMD="$TEST_CMD --disable-target-filtering"
    fi
    TEST_CMD="$TEST_CMD --target-threshold $TARGET_THRESHOLD"
    
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

# Step 4: Evaluation
EVALUATION_SUCCESS=false
if [[ "$SKIP_EVALUATION" == false ]]; then
    echo "=========================================="
    echo "Step 4: Running evaluation"
    echo "=========================================="
    
    TEST_CSV="./dataset/test_${DATASET_NAME}.csv"
    if [[ ! -f "$TEST_CSV" ]]; then
        echo "Error: Test CSV not found: $TEST_CSV"
        exit 1
    fi
    
    # Use dataset-specific output directory for evaluation
    DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
    
    # Check for model files
    MODEL_FILES=($(find "$DATASET_OUTPUT_DIR" -name "*.tar" 2>/dev/null | sort))
    if [[ ${#MODEL_FILES[@]} -eq 0 ]]; then
        echo "Error: No model checkpoint files found in $DATASET_OUTPUT_DIR"
        exit 1
    fi
    
    # Clear GPU memory before evaluation
    echo "Clearing GPU memory cache before evaluation..."
    if [ -f "gpu_memory_manager.py" ]; then
        python gpu_memory_manager.py --clear
    else
        python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || echo "Could not clear GPU cache"
    fi
    
    # Build evaluation command with memory management
    EVAL_CMD="python test.py --model \"$DATASET_OUTPUT_DIR\" --csv \"$TEST_CSV\" --batch-size $BATCH_SIZE --save-predictions --gpu-ids $GPU_IDS"
    if [[ "$DISABLE_NORMALIZATION" == true ]]; then
        EVAL_CMD="$EVAL_CMD --disable-normalization"
    fi
    
    # Add uint16 conversion flag if enabled
    if [[ "$USE_UINT16_CONVERSION" == true ]]; then
        EVAL_CMD="$EVAL_CMD --uint16-conversion"
    fi
    
    # Add clipping options
    if [[ "$ENABLE_CLIPPING" == true ]]; then
        EVAL_CMD="$EVAL_CMD --enable-clipping --clipping-threshold $CLIPPING_THRESHOLD"
    fi
    if [[ "$DISABLE_TARGET_FILTERING" == true ]]; then
        EVAL_CMD="$EVAL_CMD --disable-target-filtering"
    fi
    EVAL_CMD="$EVAL_CMD --target-threshold $TARGET_THRESHOLD"
    
    echo "Command: $EVAL_CMD"
    echo ""
    
    {
        echo "Step 4: Evaluation"
        echo "=================="
        echo "Command: $EVAL_CMD"
        echo "Start: $(date)"
        echo ""
    } >> "$PIPELINE_LOG"
    
    # Step 4a: Generate predictions (if not already exist or if forced)
    PREDICTIONS_DIR="${DATASET_OUTPUT_DIR}/predictions"
    PREDICTION_SUCCESS=false
    if [[ "$FORCE_REGENERATE_PREDICTIONS" == true ]] || [[ ! -d "$PREDICTIONS_DIR" ]] || [[ -z "$(ls -A $PREDICTIONS_DIR 2>/dev/null)" ]]; then
        if [[ "$FORCE_REGENERATE_PREDICTIONS" == true ]] && [[ -d "$PREDICTIONS_DIR" ]]; then
            echo "Force regenerating predictions, removing existing directory..."
            rm -rf "$PREDICTIONS_DIR"
        fi
        
        echo "Generating predictions for evaluation..."
        eval "$EVAL_CMD" 2>&1 | tee -a "$PIPELINE_LOG"
        
        if [[ $? -eq 0 ]]; then
            PREDICTION_SUCCESS=true
            echo "Prediction generation completed!"
        else
            echo "ERROR: Prediction generation failed!"
            exit 1
        fi
    else
        echo "Predictions directory already exists and contains files"
        echo "Skipping prediction generation. Use --force-regenerate to overwrite"
        PREDICTION_SUCCESS=true
    fi
    
    # Step 4b: Run evaluation metrics
    if [[ "$PREDICTION_SUCCESS" == true ]]; then
        echo "Running evaluation metrics..."
        METRICS_CMD="python evaluate.py --predictions-dir \"$PREDICTIONS_DIR\" --csv-file \"$TEST_CSV\" --dataset-name \"$DATASET_NAME\" --output-dir \"$DATASET_OUTPUT_DIR\""
        
        # Add clipping options
        if [[ "$ENABLE_CLIPPING" == true ]]; then
            METRICS_CMD="$METRICS_CMD --enable-clipping --clipping-threshold $CLIPPING_THRESHOLD"
        fi
        if [[ "$DISABLE_TARGET_FILTERING" == true ]]; then
            METRICS_CMD="$METRICS_CMD --disable-target-filtering"
        fi
        METRICS_CMD="$METRICS_CMD --target-threshold $TARGET_THRESHOLD"
        
        echo "Command: $METRICS_CMD"
        
        eval "$METRICS_CMD" 2>&1 | tee -a "$PIPELINE_LOG"
        
        if [[ $? -eq 0 ]]; then
            EVALUATION_SUCCESS=true
            # Count prediction files
            PRED_COUNT=$(find "$PREDICTIONS_DIR" -name "*_pred.npy" 2>/dev/null | wc -l)
            
            {
                echo ""
                echo "Evaluation completed: $(date)"
                echo "Total prediction files: $PRED_COUNT"
                echo ""
            } >> "$PIPELINE_LOG"
            
            echo "Evaluation completed!"
            echo "Total prediction files: $PRED_COUNT"
            echo "Results saved to: $DATASET_OUTPUT_DIR"
        else
            echo "ERROR: Evaluation metrics failed!"
            {
                echo ""
                echo "Evaluation failed: $(date)"
                echo ""
            } >> "$PIPELINE_LOG"
        fi
    fi
    echo ""
else
    echo "=========================================="
    echo "Step 4: Skipping evaluation"
    echo "=========================================="
    echo ""
    
    {
        echo "Step 4: Evaluation - SKIPPED"
        echo "============================"
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

if [[ "$SKIP_EVALUATION" == false ]]; then
    echo ""
    echo "Evaluation Results Summary:"
    echo "=========================="
    # Only show results if the current evaluation succeeded
    if [[ "$EVALUATION_SUCCESS" == true ]]; then
        # Show the most recent evaluation results for the current dataset only
        DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
        LATEST_EVAL_RESULTS=$(find "$DATASET_OUTPUT_DIR" -name "evaluation_results_*.txt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
        if [[ -n "$LATEST_EVAL_RESULTS" ]]; then
            echo "Latest evaluation results from: $(basename "$LATEST_EVAL_RESULTS")"
            # Show key metrics
            echo ""
            echo "=== QUICK SUMMARY ==="
            grep -E "(RMSE|MAE|δ[₁₂₃])" "$LATEST_EVAL_RESULTS" 2>/dev/null | head -7 || echo "Evaluation results saved to file"
        else
            echo "No evaluation results file found"
        fi
        
        # Show prediction count
        PREDICTIONS_DIR="${DATASET_OUTPUT_DIR}/predictions"
        if [[ -d "$PREDICTIONS_DIR" ]]; then
            PRED_COUNT=$(find "$PREDICTIONS_DIR" -name "*_pred.npy" 2>/dev/null | wc -l)
            echo "Total prediction files generated: $PRED_COUNT"
        fi
    else
        echo "Evaluation failed for dataset: $DATASET_NAME"
        echo "Check the pipeline log for details: $PIPELINE_LOG"
    fi
fi
echo "=============================================="
