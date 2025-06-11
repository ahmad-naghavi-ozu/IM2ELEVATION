import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata_dynamic
import warnings
import os

# Suppress ALL warnings for clean training output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
torch.backends.cudnn.benchmark = True  # Optimize CUDA performance

import numpy as np
import sobel
from models import modules, net, resnet, densenet, senet
import cv2
import os
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch IM2ELEVATION Dynamic Training')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

parser.add_argument('--data', default='adjust')
parser.add_argument('--csv', default='')
parser.add_argument('--model', default='')
parser.add_argument('--input_size', default=440, type=int, 
                    help='input image size (square, e.g., 256, 440, 512)')
parser.add_argument('--batch_size', default=1, type=int,
                    help='training batch size')

args = parser.parse_args()

# Extract dataset name from path for model naming
dataset_name = os.path.basename(args.data.rstrip('/'))
save_model = args.data+'/'+dataset_name+f'_dynamic_{args.input_size}_model_'
if not os.path.exists(args.data):
    os.makedirs(args.data)


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


def main():
    global args
    global best_loss

    args = parser.parse_args()
    model_dir = args.data
    
    # Print configuration
    print(f'🔧 Dynamic Training Configuration:', flush=True)
    print(f'   📁 Dataset: {args.data}', flush=True)
    print(f'   📊 CSV: {args.csv}', flush=True) 
    print(f'   📐 Input Size: {args.input_size}x{args.input_size}', flush=True)
    print(f'   📦 Batch Size: {args.batch_size}', flush=True)
    print(f'   🔢 Epochs: {args.epochs}', flush=True)
    print(f'   📈 Learning Rate: {args.lr}', flush=True)
    print('', flush=True)

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    
    if args.model == '':
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
        state_dict = torch.load(args.model)['state_dict']
        model.load_state_dict(state_dict)
    
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    # Initialize TensorBoard
    log_dir = os.path.join(args.data, f'logs_dynamic_{args.input_size}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Dynamic data loading with configurable input size
    train_loader = loaddata_dynamic.getTrainingData(
        batch_size=args.batch_size, 
        csv_data=args.csv,
        input_size=args.input_size
    )

    best_loss = float('inf')
    best_epoch = 0
    
    print(f'🚀 Starting dynamic training with input size {args.input_size}x{args.input_size}...', flush=True)
    print('', flush=True)
    
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, writer)
        
        epoch_time = time.time() - start_time
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Learning_Rate', args.lr, epoch)
        
        # Save best model only (space efficient)
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss
            best_epoch = epoch
            
            # Save best checkpoint
            best_model_name = save_model + 'best.pkl'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'input_size': args.input_size,
                'batch_size': args.batch_size,
            }, best_model_name)
            
            print(f'💾 New best model saved! Loss: {best_loss:.6f} (Epoch {epoch+1})', flush=True)
        
        # Always save latest checkpoint (overwrite previous)
        latest_model_name = save_model + 'latest.pkl'
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'input_size': args.input_size,
            'batch_size': args.batch_size,
        }, latest_model_name)
        
        # Progress display
        print(f'📈 Epoch [{epoch+1:3d}/{args.epochs}] '
              f'Loss: {train_loss:.6f} '
              f'Best: {best_loss:.6f} (E{best_epoch+1}) '
              f'Time: {epoch_time:.1f}s', 
              flush=True)
    
    writer.close()
    print('', flush=True)
    print(f'✅ Training completed!', flush=True)
    print(f'   🏆 Best Loss: {best_loss:.6f} (Epoch {best_epoch+1})', flush=True)
    print(f'   💾 Best Model: {save_model}best.pkl', flush=True)
    print(f'   📊 Input Size: {args.input_size}x{args.input_size}', flush=True)


def train(train_loader, model, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    end = time.time()
    
    for i, sample_batched in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        image, depth = sample_batched['image'], sample_batched['depth']
        depth = depth.cuda(non_blocking=True)
        image = image.cuda()
        
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        
        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()
        
        output = model(image)
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
        
        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log training metrics every 100 batches
        if i % 100 == 0:
            writer.add_scalar('Batch_Loss/Train', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss_Components/Depth', loss_depth.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss_Components/Normal', loss_normal.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss_Components/Gradient_X', loss_dx.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss_Components/Gradient_Y', loss_dy.item(), epoch * len(train_loader) + i)
   
    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


if __name__ == '__main__':
    main()
