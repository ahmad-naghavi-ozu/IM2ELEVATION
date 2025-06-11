#!/bin/bash

# Dynamic Testing Script for IM2ELEVATION
# Usage: ./run_test_dynamic.sh <dataset_dir> [options]

set -e

# Parse arguments
if [ $# -lt 1 ]; then
    echo "❌ Usage: $0 <dataset_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --single <size>        Single-scale testing (e.g., --single 440)"
    echo "  --multi               Multi-scale testing (default sizes)"
    echo "  --multi <size1,size2> Multi-scale with custom sizes"
    echo "  --model <path>        Specific model checkpoint"
    echo "  --verbose             Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 DFC2023Amini --single 440"
    echo "  $0 DFC2023Amini --multi"
    echo "  $0 DFC2023Amini --multi 256,440,512,640"
    echo "  $0 DFC2023Amini --single 512 --model path/to/model.pkl"
    exit 1
fi

DATASET_DIR="$1"
shift

# Default values
MODE="single"
INPUT_SIZE=440
MULTI_SCALES="256,320,440,512,640"
MODEL_PATH=""
VERBOSE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)
            MODE="single"
            INPUT_SIZE="$2"
            shift 2
            ;;
        --multi)
            MODE="multi"
            if [[ $# -gt 1 && $2 != --* ]]; then
                MULTI_SCALES="$2"
                shift 2
            else
                shift 1
            fi
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
            exit 1
            ;;
    esac
done

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Look for test CSV file
CSV_FILE=""
if [ -f "$DATASET_DIR/test.csv" ]; then
    CSV_FILE="$DATASET_DIR/test.csv"
elif [ -f "$DATASET_DIR/valid.csv" ]; then
    CSV_FILE="$DATASET_DIR/valid.csv"
    echo "⚠️  Using validation set for testing (no test.csv found)"
else
    echo "❌ No test CSV found in $DATASET_DIR"
    echo "   Expected: test.csv or valid.csv"
    exit 1
fi

echo "🧪 IM2ELEVATION Dynamic Testing"
echo "==============================="
echo "📁 Dataset: $DATASET_DIR"
echo "📊 CSV: $(basename $CSV_FILE)"

if [ "$MODE" = "single" ]; then
    echo "📐 Mode: Single-scale (${INPUT_SIZE}x${INPUT_SIZE})"
    
    # Validate input size for single-scale
    if ! [[ "$INPUT_SIZE" =~ ^[0-9]+$ ]] || [ "$INPUT_SIZE" -lt 128 ] || [ "$INPUT_SIZE" -gt 1024 ]; then
        echo "❌ Invalid input size: $INPUT_SIZE (must be 128-1024)"
        exit 1
    fi
    
    python test_dynamic.py \
        --data "$DATASET_DIR" \
        --csv "$CSV_FILE" \
        --input_size "$INPUT_SIZE" \
        ${MODEL_PATH:+--model "$MODEL_PATH"} \
        $VERBOSE
        
else
    echo "📐 Mode: Multi-scale"
    echo "📏 Sizes: $MULTI_SCALES"
    
    # Convert comma-separated sizes to space-separated for Python
    SCALES_ARRAY=$(echo "$MULTI_SCALES" | tr ',' ' ')
    
    python test_dynamic.py \
        --data "$DATASET_DIR" \
        --csv "$CSV_FILE" \
        --multi_scale \
        --scales $SCALES_ARRAY \
        ${MODEL_PATH:+--model "$MODEL_PATH"} \
        $VERBOSE
fi

echo ""
echo "✅ Dynamic testing completed!"
