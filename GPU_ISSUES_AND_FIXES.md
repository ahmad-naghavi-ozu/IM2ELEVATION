# GPU Utilization Problems and Solutions - IM2ELEVATION Pipeline

## 🚨 Identified Problems

### 1. **GPU Device Mismatch**
- **Issue**: You specified `--gpu-ids 2` but training/testing was allocating memory on GPU 0
- **Evidence**: Error shows "Process 2189819 has 8.85 GiB memory in use" on GPU 0, while you specified GPU 2
- **Root Cause**: Inconsistent GPU allocation between different pipeline stages

### 2. **Memory Allocation Errors**
- **Training Phase**: CUDA OOM during test evaluation within training loop
  ```
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB. 
  GPU 0 has a total capacity of 10.57 GiB of which 11.00 MiB is free.
  ```
- **Testing/Evaluation Phase**: Multiple OOM errors when trying to allocate 380MB
  ```
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 380.00 MiB. 
  GPU 2 has a total capacity of 10.57 GiB of which 146.25 MiB is free.
  ```

### 3. **Memory Fragmentation**
- **Issue**: PyTorch suggested using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Evidence**: "257.53 MiB is reserved by PyTorch but unallocated"

### 4. **Poor Memory Management Between Pipeline Steps**
- **Issue**: No memory cleanup between training → testing → evaluation
- **Result**: Accumulated memory usage causing downstream failures

## ✅ Implemented Solutions

### 1. **Fixed GPU Device Assignment**
```bash
# Before: Redundant --single-gpu option
--single-gpu --gpu-ids 2  # Confusing and redundant

# After: Automatic detection based on number of GPUs specified
--gpu-ids 2      # Automatically detected as single GPU
--gpu-ids 0,1,2  # Automatically detected as multi-GPU
```

### 2. **Added Memory Management Environment Variables**
```bash
# Added to all train/test/eval commands
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3. **Created GPU Memory Manager Utility**
- **File**: `gpu_memory_manager.py`
- **Features**:
  - Clear GPU cache: `python gpu_memory_manager.py --clear`
  - Monitor memory usage: `python gpu_memory_manager.py --monitor 30`
  - Get memory info: `python gpu_memory_manager.py --info`
  - Suggest optimal batch size: `python gpu_memory_manager.py --suggest-batch-size`
  - Kill GPU processes: `python gpu_memory_manager.py --kill-processes`

### 4. **Added Memory Cleanup Between Pipeline Steps**
```bash
# After training
python gpu_memory_manager.py --clear

# Before testing  
python gpu_memory_manager.py --clear

# Before evaluation
python gpu_memory_manager.py --clear
```

### 5. **Improved Test Script Memory Handling**
- Added `torch.no_grad()` context for inference
- Added `torch.cuda.empty_cache()` between batches
- Added explicit tensor deletion: `del image, depth, output`
- Fixed numpy save function call

### 6. **Reduced Default Batch Size**
```bash
# Before: BATCH_SIZE=2 (often causes OOM)
# After:  BATCH_SIZE=1 (safer default)
```

### 7. **Added Pre-Pipeline GPU Status Check**
- Shows GPU memory status before starting
- Suggests optimal batch sizes for available GPUs
- Helps users make informed decisions

## 🔧 Usage Examples

### Basic Pipeline with GPU Memory Management
```bash
# Check GPU status first
./gpu_memory_manager.py --info --suggest-batch-size

# Run pipeline with memory-safe settings (single GPU auto-detected)
./run_pipeline.sh --gpu-ids 2 --batch-size 1 --epochs 5

# Run pipeline with multiple GPUs (multi-GPU auto-detected)
./run_pipeline.sh --gpu-ids 0,1,2 --batch-size 1 --epochs 5

# Monitor GPU usage during training
./gpu_memory_manager.py --monitor 60
```

### Emergency Memory Recovery
```bash
# Clear all GPU memory
./gpu_memory_manager.py --clear

# Kill all GPU processes if needed
./gpu_memory_manager.py --kill-processes

# Check memory status
./gpu_memory_manager.py --info
```

### Memory-Optimized Pipeline Run
```bash
# For single GPU with limited memory (automatically detected as single GPU)
./run_pipeline.sh --gpu-ids 2 --batch-size 1

# For multiple GPUs (automatically detected as multi-GPU)
./run_pipeline.sh --gpu-ids 2,3 --batch-size 1
```

## 📊 Expected Improvements

1. **Simplified GPU Configuration**: Automatic single vs multi-GPU detection eliminates confusion
2. **Consistent GPU Usage**: All pipeline stages now use the same specified GPU(s)
3. **Reduced OOM Errors**: Memory management and smaller batch size reduce memory pressure
4. **Better Memory Utilization**: Expandable segments reduce fragmentation
5. **Monitoring Capabilities**: Real-time GPU memory monitoring available
6. **Automatic Recovery**: Memory cleanup between pipeline stages prevents accumulation

## ⚠️ Additional Recommendations

1. **Monitor Memory Usage**: Use `--monitor` during long training runs
2. **Start Small**: Begin with `--batch-size 1` and increase if memory allows
3. **Specify GPUs Clearly**: Use `--gpu-ids 2` for single GPU or `--gpu-ids 0,1,2` for multiple
4. **Regular Cleanup**: Run `gpu_memory_manager.py --clear` between experiments
5. **Check Before Training**: Always run `--info --suggest-batch-size` before starting

## 🔍 Debugging Commands

```bash
# Check what's using GPU memory
nvidia-smi

# Get detailed PyTorch memory info
python gpu_memory_manager.py --info

# Monitor real-time usage
python gpu_memory_manager.py --monitor 30

# Emergency cleanup
python gpu_memory_manager.py --kill-processes --clear
```

## Summary

The main GPU utilization problems were:

1. **Redundant GPU Options**: The `--single-gpu` option was unnecessary - the system automatically detects single vs multi-GPU based on the number of GPUs specified in `--gpu-ids`
2. **GPU Memory Leaks**: Added memory clearing between pipeline steps
3. **CUDA Memory Fragmentation**: Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. **Missing `torch.no_grad()`**: Added to testing loop to prevent gradient computation
5. **Batch Size Too Large**: Reduced default batch size from 2 to 1

### Key Improvements Made:

1. **Simplified GPU Selection**: Removed redundant `--single-gpu` option
2. **Automatic GPU Mode Detection**: Single vs multi-GPU determined by GPU count
3. **Memory Management**: Added GPU cache clearing between steps
4. **Reduced Memory Usage**: Lower default batch size and proper tensor cleanup
5. **Better Error Handling**: Added memory status monitoring and suggestions

The pipeline now automatically handles single vs multiple GPU configurations based on the `--gpu-ids` parameter, making it much simpler and less error-prone.
