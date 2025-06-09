#!/usr/bin/env python3
"""
IM2ELEVATION Environment Verification Script
Tests if the current environment can run the IM2ELEVATION project
"""

import sys
import os
import importlib

def test_import(module_name, display_name=None, required=True):
    """Test if a module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {display_name:<20} - version: {version}")
        return True
    except ImportError as e:
        status = "🔴 REQUIRED" if required else "🟡 OPTIONAL"
        print(f"{status} {display_name:<20} - NOT FOUND: {e}")
        return False

def test_torch_cuda():
    """Test PyTorch CUDA functionality"""
    try:
        import torch
        print(f"\n🧪 PyTorch CUDA Test:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        return torch.cuda.is_available()
    except Exception as e:
        print(f"❌ PyTorch CUDA test failed: {e}")
        return False

def test_file_access():
    """Test if we can access the IM2ELEVATION project files"""
    project_files = [
        'train.py',
        'test.py', 
        'loaddata.py',
        'util.py',
        'models/net.py'
    ]
    
    print(f"\n📁 Project File Access Test:")
    all_found = True
    for file_path in project_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_found = False
    
    return all_found

def main():
    print("🔬 IM2ELEVATION Environment Verification")
    print("=" * 50)
    
    # Check if we're in the right environment
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
    print(f"Current environment: {env_name}")
    
    if env_name != 'im2elevation':
        print("⚠️  Not in im2elevation environment!")
        print("Please run: conda activate im2elevation")
    
    print(f"\n📦 Package Import Tests:")
    print("-" * 30)
    
    # Core required packages
    core_results = []
    core_results.append(test_import('torch', 'PyTorch', required=True))
    core_results.append(test_import('torchvision', 'TorchVision', required=True))
    core_results.append(test_import('cv2', 'OpenCV', required=True))
    core_results.append(test_import('numpy', 'NumPy', required=True))
    core_results.append(test_import('scipy', 'SciPy', required=True))
    core_results.append(test_import('PIL', 'Pillow', required=True))
    core_results.append(test_import('matplotlib', 'Matplotlib', required=True))
    core_results.append(test_import('pandas', 'Pandas', required=True))
    
    # Optional but useful packages
    print(f"\n📦 Optional Package Tests:")
    print("-" * 30)
    optional_results = []
    optional_results.append(test_import('skimage', 'Scikit-Image', required=False))
    optional_results.append(test_import('sklearn', 'Scikit-Learn', required=False))
    optional_results.append(test_import('imageio', 'ImageIO', required=False))
    optional_results.append(test_import('osgeo.gdal', 'GDAL', required=False))
    optional_results.append(test_import('rasterio', 'Rasterio', required=False))
    optional_results.append(test_import('pytorch_ssim', 'PyTorch-SSIM', required=False))
    # Test TensorBoard (modern approach)
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard (PyTorch): Successfully imported")
        optional_results.append(True)
    except ImportError as e:
        print(f"❌ TensorBoard (PyTorch): Import failed - {e}")
        optional_results.append(False)
    
    # Legacy tensorboard_logger check (for reference)
    optional_results.append(test_import('tensorboard_logger', 'TensorBoard Logger (Legacy)', required=False))
    
    # Development tools
    print(f"\n📦 Development Tools:")
    print("-" * 30)
    dev_results = []
    dev_results.append(test_import('jupyter', 'Jupyter', required=False))
    dev_results.append(test_import('IPython', 'IPython', required=False))
    
    # CUDA test
    cuda_working = test_torch_cuda()
    
    # File access test
    files_found = test_file_access()
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print("=" * 30)
    
    core_working = sum(core_results)
    optional_working = sum(optional_results)
    dev_working = sum(dev_results)
    
    print(f"Core packages:        {core_working}/{len(core_results)}")
    print(f"Optional packages:    {optional_working}/{len(optional_results)}")
    print(f"Development tools:    {dev_working}/{len(dev_results)}")
    print(f"CUDA functionality:   {'✅' if cuda_working else '❌'}")
    print(f"Project files:        {'✅' if files_found else '❌'}")
    
    # Overall assessment
    print(f"\n🎯 ENVIRONMENT STATUS:")
    if core_working >= 7 and cuda_working and files_found:
        print("🟢 EXCELLENT - Ready for IM2ELEVATION training!")
        status = 0
    elif core_working >= 6 and files_found:
        print("🟡 GOOD - Basic functionality available, some packages missing")
        status = 1
    elif core_working >= 4:
        print("🟠 PARTIAL - Core packages available but needs work")
        status = 2
    else:
        print("🔴 POOR - Major packages missing, environment needs setup")
        status = 3
    
    print(f"\n💡 Next steps:")
    if status == 0:
        print("  Try running: python train.py --help")
    elif status <= 2:
        print("  Run: python tools/environment_setup/install_lightweight.py")
    else:
        print("  Check conda environment and reinstall packages")
    
    return status

if __name__ == "__main__":
    sys.exit(main())
