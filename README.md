# IM2ELEVATION: Building Height Estimation from Single-View Aerial Imagery

[![Paper](https://img.shields.io/badge/Paper-Open_Access-green)](https://www.mdpi.com/2072-4292/12/17/2719/pdf)
[![Dataset](https://img.shields.io/badge/Dataset-OSI-blue)](https://drive.google.com/drive/folders/14sBkjeYY7R1S9NzWI5fGLX8XTuc8puHy?usp=sharing)
[![Weights](https://img.shields.io/badge/Weights-Download-orange)](https://drive.google.com/file/d/1KZ50MQY5Fof8SAoJN34bxnk7mfou4frF/view?usp=sharing)

> **An end-to-end deep learning approach for estimating Digital Surface Models (DSM) and building heights from single-view aerial imagery using convolutional-deconvolutional neural networks.**

## 📄 Publication

This implementation is based on the work presented in:

```bibtex
@article{liu2020im2elevation,
  title={IM2ELEVATION: Building Height Estimation from Single-View Aerial Imagery},
  author={Liu, Chao-Jung and Krylov, Vladimir A and Kane, Paul and Kavanagh, Geraldine and Dahyot, Rozenn},
  journal={Remote Sensing},
  volume={12},
  number={17},
  pages={2719},
  year={2020},
  publisher={MDPI},
  doi={10.3390/rs12172719}
}
```

**Please cite this journal paper** (available as Open Access [PDF](https://www.mdpi.com/2072-4292/12/17/2719/pdf)) when using this code or dataset.

## 🎯 Overview

IM2ELEVATION addresses the challenging problem of estimating building heights and Digital Surface Models (DSMs) from single-view aerial imagery. This is an inherently ill-posed problem that we solve using deep learning techniques.

### Key Contributions

- **End-to-end trainable architecture**: Fully convolutional-deconvolutional network that learns direct mapping from RGB aerial imagery to DSM
- **Multi-sensor fusion**: Combines aerial optical and LiDAR data for training data preparation
- **Registration improvement**: Novel registration procedure using Mutual Information and Hough transform validation
- **State-of-the-art performance**: Validated on high-resolution Dublin dataset and popular DSM estimation benchmarks

## 🏗️ Architecture

The model uses a **Squeeze-and-Excitation Network (SENet-154)** as the encoder backbone with the following components:

### Network Architecture
```
Input RGB (3 channels) → SENet-154 Encoder → Multi-Feature Fusion → Refinement → DSM Output (1 channel)
```

#### Core Components:
1. **Encoder (E_senet)**: SENet-154 pretrained on ImageNet, extracts hierarchical features at 5 different scales
2. **Decoder (D2)**: Up-projection blocks with skip connections for spatial resolution recovery
3. **Multi-Feature Fusion (MFF)**: Fuses features from all encoder blocks at the same spatial resolution
4. **Refinement Module (R)**: Final convolutional layers for output refinement

#### Technical Details:
- **Input Size**: 440×440 RGB images
- **Output Size**: Variable (typically 220×220 or 440×440 for DSM)
- **Loss Function**: Combined loss with depth, gradient, and surface normal terms
- **Training**: Adam optimizer with learning rate 0.0001, batch size 2

## 📊 Loss Function

The model uses a sophisticated multi-term loss function:

```python
L_total = L_depth + L_normal + L_dx + L_dy
```

Where:
- **L_depth**: `log(|output - depth| + 0.5)` - Direct depth estimation loss
- **L_normal**: `|1 - cos(output_normal, depth_normal)|` - Surface normal consistency
- **L_dx, L_dy**: Gradient losses in x and y directions for edge preservation

## 🚀 Quick Start

### Environment Setup
```bash
# Create conda environment with modern dependencies
mamba env create -f tools/environment_setup/environment_modern.yml
conda activate im2elevation

# Install additional packages
pip install pytorch-ssim tensorboard --no-cache-dir
```

### Training
```bash
python train.py --data output_folder --csv path/to/training.csv --epochs 100
```

### Testing
```bash
python test.py --model path/to/model_folder --csv path/to/test.csv --outfile results.txt
```

## 📥 Downloads

### Pre-trained Weights
Download the trained SENet-154 model weights:
**[Download Trained Weights](https://drive.google.com/file/d/1KZ50MQY5Fof8SAoJN34bxnk7mfou4frF/view?usp=sharing)**

### OSI Dataset
High-resolution dataset captured over central Dublin, Ireland:
- **LiDAR point cloud**: 2015
- **Optical aerial images**: 2017
- **Coverage**: Central Dublin area with building height annotations

**[Download OSI Dataset](https://drive.google.com/drive/folders/14sBkjeYY7R1S9NzWI5fGLX8XTuc8puHy?usp=sharing)**

## 💡 Implementation Notes

### Data Processing Pipeline
1. **Registration**: Mutual Information-based alignment of optical and LiDAR data
2. **Validation**: Hough transform-based validation to detect registration failures
3. **Adjustment**: Interpolation-based correction of failed registration patches
4. **Augmentation**: Standard data augmentation techniques during training

### Model Variants
The implementation supports three encoder backbones:
- **ResNet-50**: `define_model(is_resnet=True, is_densenet=False, is_senet=False)`
- **DenseNet-161**: `define_model(is_resnet=False, is_densenet=True, is_senet=False)`
- **SENet-154** (Recommended): `define_model(is_resnet=False, is_densenet=False, is_senet=True)`

### Key Features
- **Multi-GPU Support**: DataParallel training on multiple GPUs
- **Skip Connections**: Enhanced feature propagation from encoder to decoder
- **Up-projection Blocks**: Learnable upsampling with feature refinement
- **Multi-scale Fusion**: Features from all encoder levels combined for rich representation

## 📁 Project Structure

```
IM2ELEVATION/
├── models/
│   ├── net.py              # Main model definition
│   ├── modules.py          # Network components (Encoder, Decoder, MFF, Refinement)
│   ├── senet.py           # SENet backbone implementation
│   ├── resnet.py          # ResNet backbone implementation
│   └── densenet.py        # DenseNet backbone implementation
├── train.py               # Training script
├── test.py                # Testing and evaluation script
├── loaddata.py           # Data loading utilities
├── util.py               # Utility functions for evaluation
├── sobel.py              # Sobel edge detection for gradient loss
├── splitGeoTiff.py       # Geospatial data processing
└── tools/
    ├── environment_setup/ # Environment setup tools
    ├── git_setup/        # Git configuration
    └── pdf_processor/    # Document processing tools
```

## 🔧 System Requirements

- **OS**: Linux (tested on Ubuntu)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Dependencies**: Python 3.11, PyTorch 2.4.0, CUDA 12.1

## 📈 Performance

The model achieves state-of-the-art performance on:
- **Dublin Dataset**: High-resolution validation with 2015 LiDAR and 2017 optical imagery
- **Standard Benchmarks**: Competitive results on popular DSM estimation datasets
- **Evaluation Metrics**: MSE, RMSE, MAE, SSIM for comprehensive assessment

## 🤝 Contributing

We welcome contributions! Please see our environment setup guide in `tools/environment_setup/` for development setup instructions.

## 📞 Contact

For questions about the implementation or dataset, please refer to the original paper or create an issue in this repository.

---

**Note**: This implementation focuses on urban building height estimation. For optimal results, ensure your aerial imagery has similar characteristics to the training data (resolution, viewing angle, urban scenes).
