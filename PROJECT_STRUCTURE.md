# IM2ELEVATION Project Structure and Work Progress Folders

## Git Branch Structure

### `master` branch
- Original IM2ELEVATION implementation
- Basic functionality and core architecture
- Fixed 440x440 input size

### `structured-datasets` branch  
- Enhanced dataset handling and CSV generation
- Bug fixes (SSIM calculation, float conversion)
- Improved training scripts with best checkpoint tracking
- Shell script automation and progress monitoring
- Integration with DFC2023Amini dataset

### `dynamic-input-sizes` branch ⭐ **Current Branch**
- Dynamic input size system (128-1024 pixels)
- Multi-scale training and testing capabilities
- Performance analysis and benchmarking tools
- Architecture compatibility without structural modifications
- Comprehensive documentation and usage guides

---

## Work Progress Folders (Excluded from Git)

### 📁 `pipeline_output/`
**Purpose**: Stores experiment results and pipeline execution logs

**Contents**:
- `pipeline_log_<dataset>_<timestamp>.txt` - Detailed training/testing logs
- `test_results_<dataset>_<timestamp>.txt` - Evaluation metrics and results
- `<dataset>/` subdirectories containing:
  - Model checkpoints (best and latest)
  - Training progress files
  - Intermediate results

**Example Structure**:
```
pipeline_output/
├── pipeline_log_DFC2023Amini_20250610_010637.txt
├── test_results_DFC2023Amini_20250610_131937.txt
└── DFC2023Amini/
    ├── DFC2023Amini_model_best_epoch_1.pth.tar
    ├── DFC2023Amini_model_latest.pth.tar
    └── training_metrics.json
```

**Why Excluded**: Large files, temporary results, user-specific paths

---

### 📁 `pretrained_model/`
**Purpose**: Stores pre-trained model weights and base encoders

**Contents**:
- `encoder/` - Pre-trained encoder weights (SENet, ResNet, DenseNet)
  - `senet154-c7b49a05.pth` - SENet154 ImageNet weights
  - `resnet50-19c8e357.pth` - ResNet50 ImageNet weights
  - `densenet161-8d451a50.pth` - DenseNet161 ImageNet weights
- Base model checkpoints for transfer learning
- Fine-tuned models from other datasets

**Example Structure**:
```
pretrained_model/
├── encoder/
│   ├── senet154-c7b49a05.pth      # 446MB
│   ├── resnet50-19c8e357.pth      # 98MB
│   └── densenet161-8d451a50.pth   # 110MB
├── nyu_depth_v2_pretrained.pkl    # NYU Depth V2 trained model
└── cityscapes_pretrained.pkl      # Cityscapes trained model
```

**Why Excluded**: Large binary files (100MB+), downloadable from public sources

---

### 📁 `runs/`
**Purpose**: TensorBoard logs and training visualization data

**Contents**:
- TensorBoard event files for training monitoring
- Loss curves and metric visualizations
- Model architecture graphs
- Hyperparameter tuning results
- Size-specific training logs for dynamic system

**Example Structure**:
```
runs/
├── experiment_DFC2023Amini_440x440/
│   ├── events.out.tfevents.1749506432.nova41.2569243.0
│   └── hparams.yaml
├── experiment_DFC2023Amini_512x512/
│   ├── events.out.tfevents.1749506543.nova41.2569244.0
│   └── scalars/
├── pipeline_output/DFC2023Amini/
│   └── events.out.tfevents.1749506432.nova41.2569243.0
└── multi_scale_comparison/
    └── comparison_results.json
```

**Usage**:
```bash
# View training progress
tensorboard --logdir=runs/

# Compare different experiments
tensorboard --logdir=runs/ --port=6006
```

**Why Excluded**: Large files, automatically generated, user-specific experiment data

---

## File Organization Best Practices

### Tracked Files (In Git)
- ✅ Source code (`.py` files)
- ✅ Configuration files (`.yml`, `.json`)
- ✅ Shell scripts (`.sh` files) 
- ✅ Documentation (`.md` files)
- ✅ Small datasets/CSVs (< 1MB)

### Excluded Files (In .gitignore)
- ❌ Large model weights (> 10MB)
- ❌ Training logs and outputs
- ❌ TensorBoard event files
- ❌ Temporary/cache files
- ❌ User-specific configurations
- ❌ Experiment results

### Storage Recommendations

1. **Local Development**: Keep all folders for active work
2. **Remote Backup**: Use separate storage for `pretrained_model/` and `pipeline_output/`
3. **Collaboration**: Share only essential results via external links
4. **Clean Workspace**: Regularly archive old experiments from `runs/`

---

## Dynamic Input Size System Files

### Core Components
- `loaddata_dynamic.py` - Configurable data loading (128-1024px)
- `train_dynamic.py` - Variable input size training
- `test_dynamic.py` - Multi-scale testing and evaluation
- `analyze_performance.py` - Memory and speed benchmarking

### Automation Scripts
- `run_train_dynamic.sh` - Dynamic training automation
- `run_test_dynamic.sh` - Dynamic testing automation  
- `run_pipeline_dynamic.sh` - Complete pipeline management

### Documentation
- `DYNAMIC_INPUT_GUIDE.md` - Comprehensive usage guide
- Input size recommendations and performance characteristics
- Architecture compatibility analysis

---

## Quick Start Commands

### Switch to Dynamic Branch
```bash
git checkout dynamic-input-sizes
```

### Train with Different Input Sizes
```bash
# Fast training (256x256)
./run_train_dynamic.sh DFC2023Amini 256 50 4 0.0005

# Standard training (440x440) 
./run_train_dynamic.sh DFC2023Amini 440 100 1 0.0001

# High-resolution training (512x512)
./run_train_dynamic.sh DFC2023Amini 512 75 1 0.0001
```

### Multi-scale Testing
```bash
# Test multiple input sizes
./run_test_dynamic.sh DFC2023Amini --multi 256,440,512,640

# Performance analysis
python analyze_performance.py --dataset DFC2023Amini
```

This structure provides clear separation between code (tracked) and work progress (excluded), enabling efficient collaboration while maintaining local experiment data.
