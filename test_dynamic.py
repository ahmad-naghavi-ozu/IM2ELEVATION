import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import loaddata_dynamic
import warnings
import os

# Suppress ALL warnings for clean testing output  
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import numpy as np
import sobel
from models import modules, net, resnet, densenet, senet
import cv2
import util
import time

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def test_single_scale(csv_file, model_path, input_size, verbose=False):
    """Test model on a single input size."""
    
    # Load model
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Check if model was trained with same input size
    model_input_size = checkpoint.get('input_size', 440)  # Default to 440 if not found
    if model_input_size != input_size and verbose:
        print(f'⚠️  Model trained on {model_input_size}x{model_input_size}, testing on {input_size}x{input_size}')
    
    cudnn.benchmark = True
    
    # Load test data with specified input size
    test_loader = loaddata_dynamic.getTestingData(
        batch_size=1, 
        csv=csv_file,
        input_size=input_size
    )
    
    model.eval()
    totalNumber = 0
    
    Ae = 0
    Pe = 0 
    Re = 0
    Fe = 0
    
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    
    if verbose:
        print(f'🧪 Testing on {input_size}x{input_size} input size...')
        print(f'📊 Total test samples: {len(test_loader)}')
    
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            image, depth = sample_batched['image'], sample_batched['depth']
            
            depth = depth.cuda(non_blocking=True)
            image = image.cuda()
            
            output = model(image)
            output = torch.nn.functional.interpolate(output, size=[depth.size(2), depth.size(3)], mode='bilinear', align_corners=True)
            
            depth_edge = edge_detection(depth)
            output_edge = edge_detection(output)
            
            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output, depth)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            edge_errors = util.evaluateError(output_edge, depth_edge)
            Ae += edge_errors['MAE']
            Pe += util.getPer(output_edge, depth_edge, 0.4)
            Re += util.getRec(output_edge, depth_edge, 0.4)
            Fe += util.getF(Pe, Re)
    
    # Calculate final metrics
    averageError = util.averageErrors(errorSum, totalNumber)
    averageError['Edge_MAE'] = Ae/totalNumber
    averageError['Edge_Precision'] = Pe/totalNumber  
    averageError['Edge_Recall'] = Re/totalNumber
    averageError['Edge_F1'] = Fe/totalNumber
    
    return averageError


def test_multi_scale(csv_file, model_path, input_sizes=[256, 320, 440, 512, 640], verbose=True):
    """Test model on multiple input sizes for comprehensive evaluation."""
    
    if verbose:
        print(f'🔬 Multi-scale Testing')
        print(f'📁 Model: {os.path.basename(model_path)}')
        print(f'📊 CSV: {os.path.basename(csv_file)}')
        print(f'📐 Input Sizes: {input_sizes}')
        print('='*60)
    
    results = {}
    
    for size in input_sizes:
        if verbose:
            print(f'\n🧪 Testing {size}x{size}...', end=' ')
            start_time = time.time()
        
        try:
            metrics = test_single_scale(csv_file, model_path, size, verbose=False)
            results[f'{size}x{size}'] = metrics
            
            if verbose:
                test_time = time.time() - start_time
                print(f'✅ Done in {test_time:.1f}s')
                print(f'   RMSE: {metrics["RMSE"]:.4f}, MAE: {metrics["MAE"]:.4f}, REL: {metrics["ABS_REL"]:.4f}')
                
        except Exception as e:
            if verbose:
                print(f'❌ Failed: {str(e)}')
            results[f'{size}x{size}'] = None
    
    if verbose:
        print('\n' + '='*60)
        print('📈 Multi-Scale Results Summary:')
        print('='*60)
        
        # Print comparison table
        print(f'{"Size":<10} {"RMSE":<8} {"MAE":<8} {"REL":<8} {"DELTA1":<8}')
        print('-' * 50)
        
        for size_name, metrics in results.items():
            if metrics:
                print(f'{size_name:<10} {metrics["RMSE"]:<8.4f} {metrics["MAE"]:<8.4f} '
                      f'{metrics["ABS_REL"]:<8.4f} {metrics["DELTA1"]:<8.4f}')
            else:
                print(f'{size_name:<10} {"FAILED":<8} {"FAILED":<8} {"FAILED":<8} {"FAILED":<8}')
    
    return results


def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()
    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)
    return edge_sobel


def find_best_checkpoint(model_dir):
    """Smart checkpoint detection."""
    checkpoints = []
    
    for file in os.listdir(model_dir):
        if file.endswith('.pkl'):
            checkpoints.append(os.path.join(model_dir, file))
    
    if not checkpoints:
        return None
    
    # Prefer 'best' over 'latest' 
    best_files = [f for f in checkpoints if 'best' in os.path.basename(f)]
    if best_files:
        return best_files[0]
    
    latest_files = [f for f in checkpoints if 'latest' in os.path.basename(f)]
    if latest_files:
        return latest_files[0]
    
    # Return most recent file
    return max(checkpoints, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description='IM2ELEVATION Dynamic Testing')
    parser.add_argument('--data', required=True, help='Dataset directory')
    parser.add_argument('--csv', required=True, help='Test CSV file')
    parser.add_argument('--model', default='', help='Model checkpoint path')
    parser.add_argument('--input_size', type=int, default=440, 
                        help='Single input size for testing')
    parser.add_argument('--multi_scale', action='store_true',
                        help='Enable multi-scale testing')
    parser.add_argument('--scales', nargs='+', type=int, 
                        default=[256, 320, 440, 512, 640],
                        help='Input sizes for multi-scale testing')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Auto-detect model if not provided
    if not args.model:
        model_path = find_best_checkpoint(args.data)
        if not model_path:
            print(f'❌ No model checkpoints found in {args.data}')
            return
        print(f'🔍 Auto-detected model: {os.path.basename(model_path)}')
    else:
        model_path = args.model
    
    if args.multi_scale:
        # Multi-scale testing
        results = test_multi_scale(args.csv, model_path, args.scales, args.verbose)
        
        # Optionally save results
        import json
        results_file = os.path.join(args.data, 'multi_scale_results.json')
        
        # Convert to JSON-serializable format
        json_results = {}
        for size, metrics in results.items():
            if metrics:
                json_results[size] = {k: float(v) for k, v in metrics.items()}
            else:
                json_results[size] = None
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f'\n💾 Results saved to: {results_file}')
        
    else:
        # Single-scale testing
        print(f'🧪 Single-Scale Testing ({args.input_size}x{args.input_size})')
        print(f'📁 Model: {os.path.basename(model_path)}')
        print(f'📊 CSV: {os.path.basename(args.csv)}')
        print('='*50)
        
        start_time = time.time()
        metrics = test_single_scale(args.csv, model_path, args.input_size, verbose=True)
        test_time = time.time() - start_time
        
        print(f'\n⏱️  Testing completed in {test_time:.1f} seconds')
        print('='*50)
        print('📊 Final Results:')
        print('='*50)
        
        for key, value in metrics.items():
            print(f'{key:<15}: {value:.6f}')


if __name__ == '__main__':
    main()
