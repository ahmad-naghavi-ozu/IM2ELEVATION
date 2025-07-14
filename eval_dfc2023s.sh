#!/bin/bash

# eval_dfc2023s.sh - Quick evaluation script for DFC2023S dataset

echo "Running evaluation for DFC2023S dataset..."
echo "This will:"
echo "1. Generate predictions using the trained model"
echo "2. Evaluate against ground truth DSM and SEM files"
echo "3. Compute all regression metrics"
echo ""

./run_eval.sh --dataset DFC2023S
