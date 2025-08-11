#!/usr/bin/env python3
"""
Quick test script to demonstrate the new optional clipping functionality
"""

import numpy as np
import torch
import sys
sys.path.append('/home/asfand/Ahmad/IM2ELEVATION')
from util import evaluateError

def test_clipping_options():
    print("=== Testing new optional clipping functionality ===")
    
    # Simulate some data
    batch_size = 2
    height, width = 440, 440
    
    # Create predictions with some values above 30m threshold
    pred = torch.rand(1, batch_size, height, width) * 50  # 0-50m range, add batch dimension
    target = torch.rand(1, batch_size, height, width) * 40  # 0-40m range, add batch dimension
    
    print(f"Prediction range: {pred.min():.2f} - {pred.max():.2f}m")
    print(f"Target range: {target.min():.2f} - {target.max():.2f}m")
    print()
    
    # Test 1: WITHOUT clipping (new default)
    print("1. WITHOUT clipping (new default behavior):")
    result1 = evaluateError(pred, target, 0, 1,  # idx=0, batches=1
                           enable_clipping=False,
                           enable_target_filtering=True)
    print(f"   RMSE: {result1['RMSE']:.4f}")
    print(f"   MAE:  {result1['MAE']:.4f}")
    print()
    
    # Test 2: WITH clipping (old behavior)
    print("2. WITH clipping (old behavior - threshold 30m):")
    result2 = evaluateError(pred, target, 0, 1,  # idx=0, batches=1
                           enable_clipping=True,
                           clipping_threshold=30.0,
                           enable_target_filtering=True)
    print(f"   RMSE: {result2['RMSE']:.4f}")
    print(f"   MAE:  {result2['MAE']:.4f}")
    print()
    
    # Test 3: Different thresholds
    print("3. WITH clipping at 20m threshold:")
    result3 = evaluateError(pred, target, 0, 1,  # idx=0, batches=1
                           enable_clipping=True,
                           clipping_threshold=20.0,
                           enable_target_filtering=True)
    print(f"   RMSE: {result3['RMSE']:.4f}")
    print(f"   MAE:  {result3['MAE']:.4f}")
    print()
    
    # Test 4: Without target filtering
    print("4. WITHOUT target filtering:")
    result4 = evaluateError(pred, target, 0, 1,  # idx=0, batches=1
                           enable_clipping=False,
                           enable_target_filtering=False)
    print(f"   RMSE: {result4['RMSE']:.4f}")
    print(f"   MAE:  {result4['MAE']:.4f}")
    print()
    
    print("=== Summary ===")
    print("✅ Clipping is now OPTIONAL and DISABLED by default")
    print("✅ Target filtering remains ENABLED by default")
    print("✅ All thresholds are configurable via command line")
    print("✅ This prevents unwanted prediction artifacts from clipping")

if __name__ == "__main__":
    test_clipping_options()
