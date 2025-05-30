# 🚀 IM2ELEVATION Environment Setup

**Fast and Simple Installation for Building Height Estimation from Aerial Imagery**

This directory contains everything you need to set up a complete, modern IM2ELEVATION environment with Python 3.11, PyTorch 2.4.0, and CUDA 12.1 support.

## 🎯 Quick Start (RECOMMENDED)

**The fastest and most reliable method - just 2 minutes!**

```bash
# 1. Create the environment (takes ~2 minutes)
mamba env create -f environment_modern.yml

# 2. Activate the environment
conda activate im2elevation

# 3. Install additional pip packages (takes ~30 seconds)
pip install pytorch-ssim tensorboard --no-cache-dir

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**That's it!** ✅ Simple, fast, and reliable.

## 💡 Why This Method Works Best

After extensive testing, we found that:
- **Complex wrapper scripts** with progress bars caused 1+ hour hangs
- **Direct mamba/conda commands** complete successfully in ~2 minutes  
- **Simple is better** than fancy progress indicators that interfere with package managers

## 📁 Essential Files

- **`environment_modern.yml`** - Modern conda environment specification (THE ESSENTIAL FILE)
- **`verify_environment.py`** - Verification script for troubleshooting

## 🎉 After Installation

1. **Activate the environment:**
   ```bash
   conda activate im2elevation
   ```

2. **Navigate to the project:**
   ```bash
   cd /home/asfand/Ahmad/IM2ELEVATION
   ```

3. **Start using IM2ELEVATION:**
   ```bash
   python train.py    # Train models
   python test.py     # Test models
   python splitGeoTiff.py  # Process geospatial data
   ```

## 🔧 System Requirements

- **OS**: Linux (tested on Ubuntu)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ free space for environment
- **Software**: Anaconda or Miniconda
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

## 📊 Environment Size

- **Environment**: ~8.5 GB (optimized)
- **All packages**: 414+ scientific computing packages
- **GPU Support**: CUDA 12.1, cuDNN included

## 🐛 Troubleshooting

If you encounter issues:

1. **Check disk space:**
   ```bash
   df -h /home
   ```

2. **Verify CUDA compatibility:**
   ```bash
   nvidia-smi
   ```

3. **Manual verification:**
   ```bash
   python verify_environment.py
   ```

4. **Clean conda cache if needed:**
   ```bash
   conda clean --all --yes
   ```

## 💡 Space Management

Clean up conda cache to free space:
```bash
conda clean --all --yes  # Frees ~3GB typically
```

Remove old environments:
```bash
conda env remove -n old_environment_name
```

## 🚀 What's Installed

**Core ML Stack:**
- Python 3.11
- PyTorch 2.4.0 with CUDA 12.1  
- NumPy, SciPy, Pandas
- Scikit-learn, Scikit-image

**Computer Vision:**
- OpenCV
- PIL/Pillow
- Matplotlib

**Geospatial:**
- GDAL (modern osgeo import)
- Rasterio

**Specialized:**
- pytorch-ssim (structural similarity)
- TensorBoard (training visualization)

## 🏆 Key Advantages

- **⚡ Fast**: Direct commands complete in ~2 minutes
- **🔬 Modern**: Python 3.11 + latest packages
- **🎮 GPU Ready**: CUDA 12.1 support included
- **🗺️ Geospatial**: Fixed GDAL imports and compatibility
- **💾 Optimized**: 8.5GB environment (vs 12GB+ alternatives)

**Ready for production deep learning research! 🚀**
