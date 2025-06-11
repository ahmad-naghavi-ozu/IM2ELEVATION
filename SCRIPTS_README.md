# IM2ELEVATION Shell Scripts

This directory contains shell scripts to streamline the IM2ELEVATION workflow with **clean output** and **smart checkpoint management**.

## 🎯 Key Improvements

### ✅ **Clean Training Output**
- **All warnings suppressed** for distraction-free training
- **Essential progress only** displayed in terminal
- **No complex monitoring scripts** needed

### 🗂️ **Smart Checkpoint Management**
- **Only saves best checkpoint** (saves 90%+ disk space)
- **Automatic cleanup** of previous checkpoints
- **Latest checkpoint** always available for resuming
- **Intelligent testing** prioritizes best models

---

## Scripts Overview

### 1. `run_train.sh` - Training Script
Handles model training with **clean output** and **best checkpoint tracking**.

**Usage:**
```bash
./run_train.sh --dataset DFC2023Amini --epochs 100 --learning-rate 0.0001 --output models_output
```

**New Features:**
- ✅ **Clean terminal output** (no warnings/verbose logs)
- 🏆 **Best checkpoint tracking** (saves only the best model)
- 🗂️ **Organized by dataset** in subdirectories
- 💾 **Space efficient** (saves ~90% disk space)

**Checkpoint Files:**
- `{dataset}_best_epoch_X.pth.tar` - Best performing checkpoint
- `{dataset}_latest.pth.tar` - Most recent checkpoint (for resuming)

### 2. `run_test.sh` - Testing Script
Evaluates model checkpoints with **smart checkpoint selection**.

**Usage:**
```bash
# Basic usage (auto-detects best checkpoint)
./run_test.sh --dataset DFC2023Amini

# Custom base directory
./run_test.sh --dataset DFC2023Amini --base-dir my_models
```

**New Features:**
- 🎯 **Prioritizes best checkpoint** automatically
- 🚀 **Faster testing** (tests only best model by default)
- 📊 **Clean output** (no warnings or verbose logs)

### 3. `run_pipeline.sh` - Full Pipeline Script
Runs the complete workflow: CSV generation → Training → Testing

**Usage:**
```bash
./run_pipeline.sh --dataset DFC2023Amini --dataset-path /home/asfand/Ahmad/datasets/DFC2023Amini --epochs 50
```

**Features:**
- 🔄 **End-to-end automation**
- ✅ **Clean output throughout**
- 🏆 **Smart checkpoint management**
- ⚡ **Efficient testing** (uses best checkpoints)

### 4. `run_progress.sh` - Training Progress Monitor
Real-time monitoring script for checking training progress from separate terminal.

**Usage:**
```bash
# Basic monitoring
./run_progress.sh

# Monitor specific dataset
./run_progress.sh MyDataset

# Monitor with custom output directory
./run_progress.sh MyDataset custom_output

# Real-time monitoring (updates every 10 seconds)
watch -n 10 ./run_progress.sh
```

**Features:**
- 📊 **Non-intrusive monitoring** (doesn't interfere with training)
- 💾 **Checkpoint counting** (shows how many models saved)
- 🏆 **Best epoch tracking** (shows which epoch performed best)
- 📁 **Disk usage monitoring** (tracks storage consumption)
- ⏱️ **Real-time updates** (perfect for long training sessions)

---

## 🚀 Quick Start Examples

### Train DFC2023Amini for 50 epochs:
```bash
./run_train.sh -d DFC2023Amini -e 50 -o my_models
# ✅ Clean output, saves only best checkpoint
```

### Test best checkpoint automatically:
```bash
./run_test.sh -d DFC2023Amini -b my_models
# 🎯 Tests only the best checkpoint automatically
```

### Full pipeline (recommended):
```bash
./run_pipeline.sh -d DFC2023Amini -p /home/asfand/Ahmad/datasets/DFC2023Amini -e 100
# 🔄 Complete workflow with clean output
```

### Resume training from latest checkpoint:
```bash
./run_train.sh -d DFC2023Amini -r 50 -m my_models/DFC2023Amini/DFC2023Amini_latest.pth.tar
```

### Monitor training progress (separate terminal):
```bash
# One-time check
./run_progress.sh

# Real-time monitoring (updates every 5 seconds)
watch -n 5 ./run_progress.sh
```

---

## 💾 Checkpoint Management

### **Before (Original Implementation):**
```
models_output/DFC2023Amini/
├── DFC2023Amini_model_0.pth.tar    # ~500MB
├── DFC2023Amini_model_1.pth.tar    # ~500MB  
├── DFC2023Amini_model_2.pth.tar    # ~500MB
├── ...
└── DFC2023Amini_model_99.pth.tar   # ~500MB
# Total: ~50GB for 100 epochs! 😱
```

### **After (Optimized Implementation):**
```
models_output/DFC2023Amini/
├── DFC2023Amini_best_epoch_42.pth.tar  # ~500MB (best performing)
└── DFC2023Amini_latest.pth.tar         # ~500MB (for resuming)
# Total: ~1GB for 100 epochs! ✅ (98% space savings)
```

### **Smart Testing:**
- **Automatic best selection:** If best checkpoint exists, test only that one
- **Fallback to all:** If no best checkpoint, test all available
- **Performance focused:** Prioritizes model quality over quantity

---

## 🎛️ Training Output Comparison

### **Before (Verbose):**
```
/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:1332: UserWarning: nn.functional.interpolate is deprecated...
/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:1452: UserWarning: Default upsampling behavior...
Loading checkpoint from: ...
Model loaded successfully...
Epoch: [0][0/12]    Time 2.456 (2.456)    Loss 0.5280 (0.5280)
...hundreds of lines...
```

### **After (Clean):**
```
Training dataset: DFC2023Amini
Starting training for 100 epochs...
============================================================
Epoch: [0][0/12]    Time 2.456 (2.456)    Loss 0.5280 (0.5280)
🏆 NEW BEST! Epoch 0, Loss: 0.485632 -> DFC2023Amini_best_epoch_0.pth.tar
Epoch: [1][0/12]    Time 2.102 (2.102)    Loss 0.4123 (0.4123)  
🏆 NEW BEST! Epoch 1, Loss: 0.398421 -> DFC2023Amini_best_epoch_1.pth.tar
Removed previous checkpoint: DFC2023Amini_best_epoch_0.pth.tar
...
============================================================
Training completed! Best model: Epoch 47, Loss: 0.234567
```

---

## 📁 Directory Structure

```
pipeline_output/
├── DFC2023Amini/
│   ├── DFC2023Amini_best_epoch_42.pth.tar  # Best checkpoint
│   ├── DFC2023Amini_latest.pth.tar         # Latest checkpoint
│   └── runs/                               # TensorBoard logs
│       └── pipeline_output/
│           └── DFC2023Amini/
└── contest/
    ├── contest_best_epoch_33.pth.tar
    └── contest_latest.pth.tar
```

---

## ⚙️ Options Reference

### Common Options:
- `-d, --dataset NAME`: Dataset name
- `-e, --epochs NUM`: Number of epochs (default: 100)
- `-lr, --learning-rate RATE`: Learning rate (default: 0.0001)
- `-o, --output DIR`: Output directory for models
- `-h, --help`: Show help message

### Training Specific:
- `-c, --csv PATH`: Custom training CSV path
- `-r, --resume EPOCH`: Resume from specific epoch
- `-m, --model PATH`: Model file for resuming

### Pipeline Specific:
- `-p, --dataset-path PATH`: Path to dataset root
- `--skip-csv`: Skip CSV generation
- `--skip-training`: Skip training  
- `--skip-testing`: Skip testing

### Testing Specific:
- `-b, --base-dir DIR`: Base directory (default: models_output)
- `--no-save`: Don't save results to file

---

## 🎯 Benefits Summary

| Feature | Before | After | Improvement |
|---------|--------|-------|-------------|
| **Disk Space** | ~50GB/100 epochs | ~1GB/100 epochs | 98% reduction |
| **Training Output** | Verbose warnings | Clean progress | Much cleaner |
| **Testing Speed** | Tests all checkpoints | Tests best only | 50x faster |
| **Organization** | Flat structure | Dataset folders | Better organized |
| **Usability** | Complex monitoring | Simple commands | Much easier |

## 📚 Help

Each script has built-in help:
```bash
./train_model.sh --help
./test_model.sh --help  
./run_pipeline.sh --help
```
