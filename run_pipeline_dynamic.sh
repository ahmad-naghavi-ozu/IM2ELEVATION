#!/bin/bash

# Dynamic Pipeline Script for IM2ELEVATION
# Comprehensive training and testing with variable input sizes
# Usage: ./run_pipeline_dynamic.sh <dataset_dir> <mode> [options]

set -e

# Function to print usage
print_usage() {
    echo "🔬 IM2ELEVATION Dynamic Pipeline"
    echo "==============================="
    echo ""
    echo "Usage: $0 <dataset_dir> <mode> [options]"
    echo ""
    echo "Modes:"
    echo "  train           Train model with specific input size"
    echo "  test            Test model with specific input size"
    echo "  multi-test      Multi-scale testing"
    echo "  compare         Compare different input sizes"
    echo "  benchmark       Comprehensive benchmark (train + test multiple sizes)"
    echo ""
    echo "Training Options:"
    echo "  --size <pixels>     Input size (e.g., 256, 440, 512)"
    echo "  --epochs <num>      Number of epochs (default: 100)"
    echo "  --batch <size>      Batch size (default: 1)"
    echo "  --lr <rate>         Learning rate (default: 0.0001)"
    echo ""
    echo "Testing Options:"
    echo "  --size <pixels>     Single input size for testing"
    echo "  --scales <list>     Comma-separated sizes for multi-test"
    echo "  --model <path>      Specific model checkpoint"
    echo ""
    echo "Examples:"
    echo "  $0 DFC2023Amini train --size 440 --epochs 50"
    echo "  $0 DFC2023Amini test --size 512"
    echo "  $0 DFC2023Amini multi-test --scales 256,440,512,640"
    echo "  $0 DFC2023Amini compare --scales 256,440,512"
    echo "  $0 DFC2023Amini benchmark --scales 256,440,512"
}

# Function to validate input size
validate_size() {
    local size=$1
    if ! [[ "$size" =~ ^[0-9]+$ ]] || [ "$size" -lt 128 ] || [ "$size" -gt 1024 ]; then
        echo "❌ Invalid input size: $size (must be 128-1024)"
        exit 1
    fi
}

# Function to estimate memory usage
estimate_memory() {
    local size=$1
    local batch=$2
    local memory_gb=$((size * size * batch * 4 / 1000000))
    echo "$memory_gb"
}

# Function to get timestamp
get_timestamp() {
    date '+%Y%m%d_%H%M%S'
}

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Parse arguments
if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

DATASET_DIR="$1"
MODE="$2"
shift 2

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Default values
INPUT_SIZE=440
EPOCHS=100
BATCH_SIZE=1
LR=0.0001
MULTI_SCALES="256,320,440,512,640"
MODEL_PATH=""
VERBOSE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            INPUT_SIZE="$2"
            validate_size "$INPUT_SIZE"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --scales)
            MULTI_SCALES="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift 1
            ;;
        *)
            echo "❌ Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Create output directory
OUTPUT_DIR="pipeline_output_dynamic"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(get_timestamp)
LOG_FILE="$OUTPUT_DIR/pipeline_log_${DATASET_DIR##*/}_${MODE}_${TIMESTAMP}.txt"

# Start logging
{
    echo "🔬 IM2ELEVATION Dynamic Pipeline"
    echo "==============================="
    echo "📅 Started: $(date)"
    echo "📁 Dataset: $DATASET_DIR"
    echo "🎯 Mode: $MODE"
    echo "📐 Input Size: ${INPUT_SIZE}x${INPUT_SIZE}"
    echo ""
} | tee "$LOG_FILE"

case $MODE in
    "train")
        log_msg "🚀 Starting dynamic training..." | tee -a "$LOG_FILE"
        
        # Memory check
        MEMORY_GB=$(estimate_memory "$INPUT_SIZE" "$BATCH_SIZE")
        log_msg "💾 Estimated GPU Memory: ~${MEMORY_GB}GB" | tee -a "$LOG_FILE"
        
        if [ "$MEMORY_GB" -gt 10 ]; then
            log_msg "⚠️  High memory usage detected. Consider reducing batch size." | tee -a "$LOG_FILE"
        fi
        
        # Start training
        ./run_train_dynamic.sh "$DATASET_DIR" "$INPUT_SIZE" "$EPOCHS" "$BATCH_SIZE" "$LR" 2>&1 | tee -a "$LOG_FILE"
        
        log_msg "✅ Training completed!" | tee -a "$LOG_FILE"
        ;;
        
    "test")
        log_msg "🧪 Starting single-scale testing..." | tee -a "$LOG_FILE"
        
        ./run_test_dynamic.sh "$DATASET_DIR" --single "$INPUT_SIZE" ${MODEL_PATH:+--model "$MODEL_PATH"} $VERBOSE 2>&1 | tee -a "$LOG_FILE"
        
        log_msg "✅ Testing completed!" | tee -a "$LOG_FILE"
        ;;
        
    "multi-test")
        log_msg "🔬 Starting multi-scale testing..." | tee -a "$LOG_FILE"
        
        ./run_test_dynamic.sh "$DATASET_DIR" --multi "$MULTI_SCALES" ${MODEL_PATH:+--model "$MODEL_PATH"} $VERBOSE 2>&1 | tee -a "$LOG_FILE"
        
        log_msg "✅ Multi-scale testing completed!" | tee -a "$LOG_FILE"
        ;;
        
    "compare")
        log_msg "📊 Starting input size comparison..." | tee -a "$LOG_FILE"
        
        IFS=',' read -ra SIZES <<< "$MULTI_SCALES"
        
        echo "🔍 Comparing performance across different input sizes:" | tee -a "$LOG_FILE"
        echo "Scales: ${SIZES[*]}" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        # Create comparison results file
        COMPARE_FILE="$OUTPUT_DIR/size_comparison_${DATASET_DIR##*/}_${TIMESTAMP}.txt"
        
        {
            echo "Input Size Comparison Results"
            echo "============================"
            echo "Dataset: $DATASET_DIR"
            echo "Timestamp: $(date)"
            echo ""
            printf "%-10s %-10s %-10s %-10s %-10s\n" "Size" "RMSE" "MAE" "ABS_REL" "DELTA1"
            echo "----------------------------------------------------"
        } > "$COMPARE_FILE"
        
        for size in "${SIZES[@]}"; do
            log_msg "Testing ${size}x${size}..." | tee -a "$LOG_FILE"
            
            # Run single test and extract key metrics
            TEST_OUTPUT=$(./run_test_dynamic.sh "$DATASET_DIR" --single "$size" ${MODEL_PATH:+--model "$MODEL_PATH"} 2>/dev/null || echo "FAILED")
            
            if [[ "$TEST_OUTPUT" == "FAILED" ]]; then
                printf "%-10s %-10s %-10s %-10s %-10s\n" "${size}x${size}" "FAILED" "FAILED" "FAILED" "FAILED" >> "$COMPARE_FILE"
            else
                # Extract metrics (this would need to be adapted based on actual output format)
                echo "$TEST_OUTPUT" | tail -10 >> "$COMPARE_FILE"
            fi
        done
        
        log_msg "📋 Comparison results saved to: $COMPARE_FILE" | tee -a "$LOG_FILE"
        ;;
        
    "benchmark")
        log_msg "🏁 Starting comprehensive benchmark..." | tee -a "$LOG_FILE"
        
        IFS=',' read -ra SIZES <<< "$MULTI_SCALES"
        BENCHMARK_DIR="$OUTPUT_DIR/benchmark_${DATASET_DIR##*/}_${TIMESTAMP}"
        mkdir -p "$BENCHMARK_DIR"
        
        echo "📊 Benchmark Configuration:" | tee -a "$LOG_FILE"
        echo "   Sizes: ${SIZES[*]}" | tee -a "$LOG_FILE"
        echo "   Epochs: $EPOCHS" | tee -a "$LOG_FILE"
        echo "   Batch Size: $BATCH_SIZE" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        for size in "${SIZES[@]}"; do
            log_msg "🔄 Benchmarking ${size}x${size}..." | tee -a "$LOG_FILE"
            
            # Train model for this size
            log_msg "  📚 Training..." | tee -a "$LOG_FILE"
            TRAIN_START=$(date +%s)
            
            ./run_train_dynamic.sh "$DATASET_DIR" "$size" "$EPOCHS" "$BATCH_SIZE" "$LR" > "$BENCHMARK_DIR/train_${size}.log" 2>&1
            
            TRAIN_END=$(date +%s)
            TRAIN_TIME=$((TRAIN_END - TRAIN_START))
            
            # Test the trained model
            log_msg "  🧪 Testing..." | tee -a "$LOG_FILE"
            TEST_START=$(date +%s)
            
            ./run_test_dynamic.sh "$DATASET_DIR" --single "$size" > "$BENCHMARK_DIR/test_${size}.log" 2>&1
            
            TEST_END=$(date +%s)
            TEST_TIME=$((TEST_END - TEST_START))
            
            log_msg "  ✅ ${size}x${size} completed (Train: ${TRAIN_TIME}s, Test: ${TEST_TIME}s)" | tee -a "$LOG_FILE"
        done
        
        # Generate benchmark summary
        SUMMARY_FILE="$BENCHMARK_DIR/benchmark_summary.txt"
        {
            echo "Benchmark Summary"
            echo "================"
            echo "Dataset: $DATASET_DIR"
            echo "Timestamp: $(date)"
            echo "Sizes tested: ${SIZES[*]}"
            echo "Epochs per size: $EPOCHS"
            echo "Batch size: $BATCH_SIZE"
            echo ""
            echo "Individual results available in:"
            for size in "${SIZES[@]}"; do
                echo "  - train_${size}.log"
                echo "  - test_${size}.log"
            done
        } > "$SUMMARY_FILE"
        
        log_msg "🏆 Benchmark completed! Results in: $BENCHMARK_DIR" | tee -a "$LOG_FILE"
        ;;
        
    *)
        echo "❌ Unknown mode: $MODE" | tee -a "$LOG_FILE"
        print_usage
        exit 1
        ;;
esac

# Final log entry
{
    echo ""
    echo "📅 Completed: $(date)"
    echo "📋 Log file: $LOG_FILE"
} | tee -a "$LOG_FILE"

echo ""
echo "✅ Dynamic pipeline completed successfully!"
echo "📋 Full log available at: $LOG_FILE"
