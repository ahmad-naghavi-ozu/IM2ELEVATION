# IM2ELEVATION Evaluation Scripts Documentation

## Overview

This document explains the evaluation pipeline and the differences between the various Python scripts and shell scripts.

## Python Scripts

### 1. `evaluate.py` (Enhanced Original)
**Purpose**: Core evaluation script that contains the `HeightRegressionMetrics` class with all regression metrics calculations.

**Key Features**:
- Original metric calculation logic preserved (mse, rmse, rmse_building, high_rise_rmse, mid_rise_rmse, low_rise_rmse, mae, delta1, delta2, delta3)
- Enhanced with new method `evaluate_from_saved_predictions()` to load and evaluate .npy prediction files
- Can load ground truth DSM and SEM files from CSV
- Handles image resizing when prediction and ground truth dimensions don't match
- Can be used as standalone script or imported as module
- Maintains backward compatibility with legacy functions

**Usage**:
```bash
python evaluate.py --predictions-dir path/to/predictions --csv-file dataset/test_dataset.csv --dataset-name dataset --output-dir path/to/results
```

### 2. `test.py` (Enhanced)
**Purpose**: Core testing script that evaluates trained models and can optionally save predictions during inference.

**Key Features**:
- Loads trained IM2ELEVATION model
- Runs inference on test dataset 
- Calculates standard testing metrics (loss, MSE, RMSE, MAE, SSIM)
- **Enhanced**: Can save prediction outputs as .npy files when `--save-predictions` flag is used
- Supports both single-GPU and multi-GPU configurations
- Automatically selects best checkpoint for evaluation

**Usage**:
```bash
# Standard testing (backward compatible)
python test.py --model path/to/model --csv dataset/test_dataset.csv

# Enhanced testing with prediction saving
python test.py --model path/to/model --csv dataset/test_dataset.csv --save-predictions --gpu-ids 0,1
```

## Shell Scripts

### 1. `run_eval.sh` (Complete Evaluation Pipeline)
**Purpose**: Full evaluation pipeline script that orchestrates both prediction generation and metric evaluation.

**Features**:
- Command-line argument parsing (like `run_pipeline.sh`)
- Generates predictions using enhanced `test.py` with `--save-predictions` flag
- Evaluates metrics using `evaluate.py`
- Supports force regeneration of predictions
- GPU configuration options
- Progress tracking and error handling

**Usage**:
```bash
./run_eval.sh --dataset DFC2023S --model-path pipeline_output/DFC2023S --gpu-ids 0,1
```

### 2. `eval_dfc2023s.sh` (Quick Shortcut)
**Purpose**: Simple wrapper script for quick evaluation of DFC2023S dataset with default settings.

**Usage**:
```bash
./eval_dfc2023s.sh
```

## Evaluation Pipeline Flow

1. **Prediction Generation** (`test.py` with `--save-predictions`):
   - Loads trained model checkpoint
   - Processes test images through model
   - Saves predictions as .npy files (440x440 pixels after interpolation)

2. **Metric Evaluation** (`evaluate.py`):
   - Loads saved prediction files
   - Loads corresponding ground truth DSM and SEM files
   - Resizes predictions to match ground truth dimensions if needed
   - Computes all regression metrics using original calculation logic
   - Saves results to timestamped text file

## Key Differences from Original Approach

### What Changed:
- **Prediction Storage**: Predictions are now saved as .npy files for analysis
- **Flexible Evaluation**: Can evaluate predictions independently of model inference
- **Better Organization**: Separate prediction generation and evaluation phases
- **Enhanced Shell Scripts**: Proper argument parsing and configuration options

### What Stayed the Same:
- **Metric Calculations**: Original regression metric logic preserved exactly
- **Model Architecture**: No changes to model or inference process
- **Data Loading**: Same data loading and preprocessing pipeline
- **GPU Support**: Same multi-GPU and single-GPU options

## Addressing IM2ELEVATION Pipeline Inconsistencies

The evaluation pipeline helps address the inconsistencies you mentioned:

1. **Input Size Handling**: Predictions are saved at model output resolution (440x440) and resized to match ground truth during evaluation
2. **Training vs Testing**: By saving predictions separately, we can analyze the actual model outputs and compare them directly with ground truth
3. **Consistent Metrics**: All metrics are computed using the same ground truth data and prediction format

## Example Workflow

```bash
# Option 1: Complete pipeline (recommended)
./run_eval.sh --dataset DFC2023S --gpu-ids 0,1

# Option 2: Manual steps
python test.py --model pipeline_output/DFC2023S --csv dataset/test_DFC2023S.csv --save-predictions
python evaluate.py --predictions-dir pipeline_output/DFC2023S/predictions --csv-file dataset/test_DFC2023S.csv --dataset-name DFC2023S

# Option 3: Quick evaluation for DFC2023S
./eval_dfc2023s.sh
```
