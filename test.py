import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import glob 
from models import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel
import argparse
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import csv
import re
import warnings

# Suppress ALL warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Reduce CUDA verbosity

import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)

    parser = argparse.ArgumentParser()
   
  
    parser.add_argument("--model")
    parser.add_argument("--csv")
    parser.add_argument("--outfile")
    parser.add_argument('--gpu-ids', default='0,1,2,3', type=str,
                        help='comma-separated list of GPU IDs to use (default: 0,1,2,3)')
    parser.add_argument('--single-gpu', action='store_true',
                        help='use only single GPU (GPU 0)')
    parser.add_argument('--batch-size', default=3, type=int,
                        help='batch size for testing (default: 3)')
    args = parser.parse_args()
    
    # Configure GPU usage
    if args.single_gpu:
        device_ids = [0]
        print(f"Using single GPU: {device_ids[0]}")
    else:
        device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        print(f"Using multiple GPUs: {device_ids}")
    
    # Check if specified GPUs are available
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if max(device_ids) >= available_gpus:
        print(f"Warning: Requested GPU {max(device_ids)} not available. Using available GPUs: {list(range(available_gpus))}")
        device_ids = list(range(min(len(device_ids), available_gpus)))

    md = glob.glob(args.model+'/*.tar')
    
    if not md:
        print("No checkpoint files found!")
        return
    
    # Prioritize best checkpoint for evaluation
    best_checkpoints = [x for x in md if 'best_epoch_' in x]
    
    if best_checkpoints:
        # Sort by epoch number and take the latest best
        best_checkpoints.sort(key=lambda x: int(x.split('best_epoch_')[1].split('.')[0]))
        selected_checkpoint = best_checkpoints[-1]
        print(f"Found best checkpoint, evaluating: {os.path.basename(selected_checkpoint)}")
    else:
        # Fallback to latest if no best checkpoint found
        latest_checkpoints = [x for x in md if 'latest' in x]
        if latest_checkpoints:
            selected_checkpoint = latest_checkpoints[0]
            print(f"No best checkpoint found, using latest: {os.path.basename(selected_checkpoint)}")
        else:
            # Use any available checkpoint
            selected_checkpoint = sorted(md, key=natural_keys)[-1]
            print(f"Using available checkpoint: {os.path.basename(selected_checkpoint)}")
    
    print("=" * 60)
    
    checkpoint_name = os.path.basename(selected_checkpoint)
    
    # Create model with suppressed output
    f = StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            print(f"Model wrapped with DataParallel using GPUs: {device_ids}")
        else:
            model = model.cuda()
            print(f"Model moved to single GPU: {device_ids[0]}")
        state_dict = torch.load(selected_checkpoint, map_location='cuda')['state_dict']
        
        # Load state dict quietly without printing parameter names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.load_state_dict(state_dict, strict=False)

    test_loader = loaddata.getTestingData(args.batch_size,args.csv)
    result = test(test_loader, model, args, checkpoint_name)
    
    print("=" * 60)



def test(test_loader, model, args, checkpoint_name=""):
    
    losses = AverageMeter()
    model.eval()
    model.cuda()
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'MAE': 0,'SSIM':0}

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']
        depth = depth.cuda(non_blocking=True)
        image = image.cuda()
        output = model(image)

        output = torch.nn.functional.interpolate(output,size=(440,440),mode='bilinear')

        batchSize = depth.size(0)
        testing_loss(depth,output,losses,batchSize)

        totalNumber = totalNumber + batchSize

        errors = util.evaluateError(output, depth,i,batchSize)

        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)
     

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    loss = float(losses.avg)

    # Enhanced output with checkpoint identification
    checkpoint_info = f" ({checkpoint_name})" if checkpoint_name else ""
    print(f'Results{checkpoint_info}:')
    print('  Loss: {loss:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | SSIM: {ssim:.4f}'.format(
        loss=loss, mse=averageError['MSE'], rmse=averageError['RMSE'], 
        mae=averageError['MAE'], ssim=averageError['SSIM']))
    
    return {
        'checkpoint': checkpoint_name,
        'loss': loss,
        'MSE': averageError['MSE'],
        'RMSE': averageError['RMSE'], 
        'MAE': averageError['MAE'],
        'SSIM': averageError['SSIM']
    }






def testing_loss(depth , output, losses, batchSize):
    
    ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
    get_gradient = sobel.Sobel().cuda()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    loss = loss_depth + loss_normal + (loss_dx + loss_dy)
    losses.update(loss.item(), batchSize)





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
        original_model = senet.senet154(pretrained=None)
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]  


if __name__ == '__main__':
    main()
