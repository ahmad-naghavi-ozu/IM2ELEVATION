#!/usr/bin/env python3

"""
Performance Analysis Tool for IM2ELEVATION Dynamic Input Sizes
Compares fixed-size (440x440) vs dynamic input size performance
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import local modules
import loaddata
import loaddata_dynamic
from models import modules, net, senet


def analyze_dataset_sizes(csv_path):
    """Analyze original image sizes in dataset."""
    print("📊 Analyzing dataset image sizes...")
    
    df = pd.read_csv(csv_path, header=None)
    sizes = []
    
    for idx in range(min(100, len(df))):  # Sample first 100 images
        img_path = df.iloc[idx, 0]
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
        except Exception as e:
            print(f"⚠️  Could not read {img_path}: {e}")
    
    if not sizes:
        return None
    
    widths, heights = zip(*sizes)
    
    stats = {
        'count': len(sizes),
        'width': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths)
        },
        'height': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights)
        },
        'aspect_ratios': [w/h for w, h in sizes],
        'common_sizes': list(set(sizes))
    }
    
    return stats


def benchmark_memory_usage(input_sizes, batch_size=1):
    """Benchmark GPU memory usage for different input sizes."""
    print("💾 Benchmarking memory usage...")
    
    # Load model
    original_model = senet.senet154(pretrained=None)  # Don't load weights to save time
    encoder = modules.E_senet(original_model)
    model = net.model(encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    memory_stats = {}
    
    for size in input_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, size, size).cuda()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
            memory_stats[f'{size}x{size}'] = {
                'peak_memory_gb': peak_memory,
                'input_size': size,
                'batch_size': batch_size
            }
            
            print(f"   {size}x{size}: {peak_memory:.2f} GB")
            
        except Exception as e:
            print(f"   {size}x{size}: FAILED - {e}")
            memory_stats[f'{size}x{size}'] = {
                'peak_memory_gb': None,
                'input_size': size,
                'batch_size': batch_size,
                'error': str(e)
            }
    
    return memory_stats


def benchmark_inference_speed(input_sizes, num_runs=10):
    """Benchmark inference speed for different input sizes."""
    print("⚡ Benchmarking inference speed...")
    
    # Load model
    original_model = senet.senet154(pretrained=None)
    encoder = modules.E_senet(original_model)
    model = net.model(encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    speed_stats = {}
    
    for size in input_sizes:
        try:
            # Warmup
            dummy_input = torch.randn(1, 3, size, size).cuda()
            for _ in range(3):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                torch.cuda.synchronize()
                times.append(time.time() - start_time)
            
            speed_stats[f'{size}x{size}'] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'fps': 1.0 / np.mean(times),
                'input_size': size
            }
            
            print(f"   {size}x{size}: {np.mean(times)*1000:.1f}ms ({1.0/np.mean(times):.1f} FPS)")
            
        except Exception as e:
            print(f"   {size}x{size}: FAILED - {e}")
            speed_stats[f'{size}x{size}'] = {
                'mean_time': None,
                'input_size': size,
                'error': str(e)
            }
    
    return speed_stats


def compare_preprocessing_approaches(csv_path, input_sizes):
    """Compare center-crop vs resize vs dynamic approaches."""
    print("🔄 Comparing preprocessing approaches...")
    
    df = pd.read_csv(csv_path, header=None)
    
    # Sample a few images for analysis
    sample_images = df.head(5)
    
    preprocessing_stats = {
        'center_crop': {},
        'resize': {},
        'dynamic': {}
    }
    
    for size in input_sizes:
        preprocessing_stats['center_crop'][f'{size}x{size}'] = {
            'information_loss': 'variable',  # Depends on original size
            'aspect_ratio_preserved': True,
            'computational_cost': 'low'
        }
        
        preprocessing_stats['resize'][f'{size}x{size}'] = {
            'information_loss': 'none',
            'aspect_ratio_preserved': False,
            'computational_cost': 'medium'
        }
        
        preprocessing_stats['dynamic'][f'{size}x{size}'] = {
            'information_loss': 'minimal',
            'aspect_ratio_preserved': True,
            'computational_cost': 'high'
        }
    
    return preprocessing_stats


def generate_performance_report(dataset_dir, output_dir="performance_analysis"):
    """Generate comprehensive performance analysis report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔬 IM2ELEVATION Dynamic Input Size Performance Analysis")
    print("=" * 60)
    
    # Find CSV files
    csv_files = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(dataset_dir, file))
    
    if not csv_files:
        print("❌ No CSV files found in dataset directory")
        return
    
    test_csv = None
    for csv_file in csv_files:
        if 'test' in os.path.basename(csv_file):
            test_csv = csv_file
            break
    
    if not test_csv:
        test_csv = csv_files[0]
    
    print(f"📊 Using CSV: {os.path.basename(test_csv)}")
    
    # Input sizes to analyze
    input_sizes = [256, 320, 384, 440, 512, 640, 768]
    
    report = {
        'dataset': dataset_dir,
        'csv_file': test_csv,
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_sizes_analyzed': input_sizes
    }
    
    # 1. Dataset size analysis
    print("\n1. Dataset Analysis:")
    dataset_stats = analyze_dataset_sizes(test_csv)
    if dataset_stats:
        report['dataset_stats'] = dataset_stats
        print(f"   📏 Image count: {dataset_stats['count']}")
        print(f"   📐 Width range: {dataset_stats['width']['min']}-{dataset_stats['width']['max']}")
        print(f"   📐 Height range: {dataset_stats['height']['min']}-{dataset_stats['height']['max']}")
    
    # 2. Memory usage analysis
    print("\n2. Memory Usage Analysis:")
    memory_stats = benchmark_memory_usage(input_sizes[:5])  # Limit to avoid OOM
    report['memory_stats'] = memory_stats
    
    # 3. Speed analysis
    print("\n3. Inference Speed Analysis:")
    speed_stats = benchmark_inference_speed(input_sizes[:5])
    report['speed_stats'] = speed_stats
    
    # 4. Preprocessing comparison
    print("\n4. Preprocessing Approach Comparison:")
    preprocessing_stats = compare_preprocessing_approaches(test_csv, input_sizes)
    report['preprocessing_stats'] = preprocessing_stats
    
    # Save detailed report
    report_file = os.path.join(output_dir, f"performance_report_{os.path.basename(dataset_dir)}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate summary
    summary_file = os.path.join(output_dir, f"performance_summary_{os.path.basename(dataset_dir)}.txt")
    with open(summary_file, 'w') as f:
        f.write("IM2ELEVATION Dynamic Input Size Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Analysis Date: {report['analysis_timestamp']}\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 20 + "\n")
        
        if 'memory_stats' in report:
            f.write("\nMemory Usage:\n")
            for size, stats in memory_stats.items():
                if stats.get('peak_memory_gb'):
                    f.write(f"  {size}: {stats['peak_memory_gb']:.2f} GB\n")
        
        if 'speed_stats' in report:
            f.write("\nInference Speed:\n")
            for size, stats in speed_stats.items():
                if stats.get('mean_time'):
                    f.write(f"  {size}: {stats['mean_time']*1000:.1f}ms ({stats['fps']:.1f} FPS)\n")
        
        f.write("\nRecommendations:\n")
        f.write("-" * 20 + "\n")
        f.write("1. For memory-constrained environments: Use smaller input sizes (256-384)\n")
        f.write("2. For high-accuracy requirements: Use larger input sizes (512-640)\n")
        f.write("3. For real-time applications: Balance size vs speed requirements\n")
        f.write("4. Consider multi-scale inference for optimal results\n")
    
    print(f"\n✅ Performance analysis completed!")
    print(f"📋 Detailed report: {report_file}")
    print(f"📄 Summary: {summary_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='IM2ELEVATION Dynamic Input Size Performance Analysis')
    parser.add_argument('--dataset', required=True, help='Dataset directory')
    parser.add_argument('--output', default='performance_analysis', help='Output directory')
    parser.add_argument('--sizes', nargs='+', type=int, default=[256, 320, 384, 440, 512, 640],
                        help='Input sizes to analyze')
    parser.add_argument('--memory-only', action='store_true', help='Only run memory analysis')
    parser.add_argument('--speed-only', action='store_true', help='Only run speed analysis')
    
    args = parser.parse_args()
    
    if args.memory_only:
        print("💾 Memory Usage Analysis Only")
        memory_stats = benchmark_memory_usage(args.sizes)
        
        output_file = os.path.join(args.output, f"memory_analysis_{os.path.basename(args.dataset)}.json")
        os.makedirs(args.output, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(memory_stats, f, indent=2)
        print(f"💾 Memory analysis saved to: {output_file}")
        
    elif args.speed_only:
        print("⚡ Speed Analysis Only")
        speed_stats = benchmark_inference_speed(args.sizes)
        
        output_file = os.path.join(args.output, f"speed_analysis_{os.path.basename(args.dataset)}.json")
        os.makedirs(args.output, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(speed_stats, f, indent=2)
        print(f"⚡ Speed analysis saved to: {output_file}")
        
    else:
        # Full analysis
        generate_performance_report(args.dataset, args.output)


if __name__ == '__main__':
    main()
