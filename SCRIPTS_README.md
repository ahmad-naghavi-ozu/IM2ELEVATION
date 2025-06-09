# IM2ELEVATION Shell Scripts

This directory contains three shell scripts to streamline the IM2ELEVATION workflow:

## Scripts Overview

### 1. `train_model.sh` - Training Script
Handles model training with configurable parameters. **Models are automatically organized by dataset name** in subdirectories.

**Usage:**
```bash
./train_model.sh --dataset DFC2023Amini --epochs 100 --learning-rate 0.0001 --output models_output
```

**Directory Structure:**
- Models saved to: `{output_dir}/{dataset_name}/`
- Example: `models_output/DFC2023Amini/DFC2023Amini_model_0.pth.tar`

**Key Options:**
- `-d, --dataset NAME`: Dataset name
- `-e, --epochs NUM`: Number of epochs (default: 100)  
- `-lr, --learning-rate RATE`: Learning rate (default: 0.0001)
- `-o, --output DIR`: Output directory for models
- `-c, --csv PATH`: Custom training CSV path
- `-r, --resume EPOCH`: Resume from specific epoch
- `-m, --model PATH`: Model file for resuming

### 2. `test_model.sh` - Testing Script
Evaluates all model checkpoints on test data. **Automatically detects dataset-specific model directories**.

**Usage:**
```bash
# Basic usage (auto-detects models_output/DFC2023Amini/)
./test_model.sh --dataset DFC2023Amini

# Custom base directory
./test_model.sh --dataset DFC2023Amini --base-dir my_models

# Specific model directory
./test_model.sh --model-dir models_output/DFC2023Amini_experiment1
```

**Key Options:**
- `-d, --dataset NAME`: Dataset name
- `-b, --base-dir DIR`: Base directory containing dataset subdirectories (default: models_output)
- `-m, --model-dir DIR`: Specific model directory (overrides auto-detection)
- `-c, --csv PATH`: Custom test CSV path
- `-o, --output FILE`: Output file for results
- `--no-save`: Don't save results (terminal only)

### 3. `run_pipeline.sh` - Full Pipeline Script
Runs the complete workflow: CSV generation → Training → Testing

**Usage:**
```bash
./run_pipeline.sh --dataset DFC2023Amini --dataset-path /home/asfand/Ahmad/datasets/DFC2023Amini --epochs 50
```

**Key Options:**
- `-d, --dataset NAME`: Dataset name
- `-p, --dataset-path PATH`: Path to dataset root
- `-e, --epochs NUM`: Number of epochs
- `-lr, --learning-rate RATE`: Learning rate
- `-o, --output DIR`: Output directory
- `--skip-csv`: Skip CSV generation
- `--skip-training`: Skip training
- `--skip-testing`: Skip testing

## Quick Start Examples

### Train DFC2023Amini for 50 epochs:
```bash
./train_model.sh -d DFC2023Amini -e 50 -o my_models
# Models saved to: my_models/DFC2023Amini/
```

### Test all checkpoints:
```bash
./test_model.sh -d DFC2023Amini -b my_models
# Tests models in: my_models/DFC2023Amini/
```

### Full pipeline (recommended):
```bash
./run_pipeline.sh -d DFC2023Amini -p /home/asfand/Ahmad/datasets/DFC2023Amini -e 100
# Models saved to: pipeline_output/DFC2023Amini/
```

### Resume training from epoch 30:
```bash
./train_model.sh -d DFC2023Amini -r 30 -m my_models/DFC2023Amini/DFC2023Amini_model_29.pth.tar
```

### Test only (skip CSV and training):
```bash
./run_pipeline.sh -d DFC2023Amini --skip-csv --skip-training
```

## Directory Structure

**Before (old structure):**
```
models_output/
├── DFC2023Amini_model_0.pth.tar
├── DFC2023Amini_model_1.pth.tar
├── contest_model_0.pth.tar
└── contest_model_1.pth.tar
```

**After (new structure):**
```
models_output/
├── DFC2023Amini/
│   ├── DFC2023Amini_model_0.pth.tar
│   ├── DFC2023Amini_model_1.pth.tar
│   └── training_log_DFC2023Amini_20250609_143022.txt
└── contest/
    ├── contest_model_0.pth.tar
    ├── contest_model_1.pth.tar
    └── training_log_contest_20250609_151045.txt
```

## Features

- **Validation**: Scripts validate inputs and show helpful error messages
- **Logging**: All operations are logged with timestamps
- **Confirmation**: Interactive confirmation before starting long operations
- **Auto-detection**: CSV paths auto-detected based on dataset name
- **Flexible**: Skip individual pipeline steps as needed
- **Progress**: Real-time output with progress information

## Output Structure

Each script creates organized output:

```
output_directory/
├── *_model_0.pth.tar          # Model checkpoints
├── *_model_1.pth.tar
├── ...
├── training_log_*.txt         # Training logs
├── test_results_*.txt         # Test results
└── pipeline_log_*.txt         # Pipeline logs
```

## Help

Each script has built-in help:
```bash
./train_model.sh --help
./test_model.sh --help  
./run_pipeline.sh --help
```
