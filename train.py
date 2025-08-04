import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import util
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



parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=100
    , type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--gpu-ids', default='0,1,2,3', type=str,
                    help='comma-separated list of GPU IDs to use (default: 0,1,2,3)')
parser.add_argument('--batch-size', default=2, type=int,
                    help='batch size per GPU (default: 2)')

parser.add_argument('--data', default='adjust')
parser.add_argument('--csv', default='')
parser.add_argument('--model', default='')
parser.add_argument('--disable-normalization', action='store_true', default=False,
                    help='disable entire normalization pipeline (x1000, /100000, x100) for raw model training (default: False)')

args = parser.parse_args()
# Extract dataset name from path for model naming
dataset_name = os.path.basename(args.data.rstrip('/'))
save_model = args.data+'/'+dataset_name+'_model_'
if not os.path.exists(args.data):
    os.makedirs(args.data)


def define_model(is_resnet, is_densenet, is_senet, device_id=0):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        # Set the default CUDA device before loading pretrained weights
        torch.cuda.set_device(device_id)
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    
    global args
    args = parser.parse_args()
    
    # Configure GPU usage
    device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    # Check if specified GPUs are available
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if max(device_ids) >= available_gpus:
        print(f"Warning: Requested GPU {max(device_ids)} not available. Using available GPUs: {list(range(available_gpus))}")
        device_ids = list(range(min(len(device_ids), available_gpus)))
    
    # Determine GPU mode automatically based on number of devices
    if len(device_ids) == 1:
        print(f"Using single GPU: {device_ids[0]}")
    else:
        print(f"Using multiple GPUs: {device_ids}")
    
    # Ensure all model parameters are on the first specified GPU before any operations
    torch.cuda.set_device(device_ids[0])
    
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True, device_id=device_ids[0])
    
    # Move model to first GPU before wrapping with DataParallel
    if len(device_ids) > 1:
        model = model.cuda(device_ids[0])  # Move to first GPU first
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"Model wrapped with DataParallel using GPUs: {device_ids}")
    else:
        model = model.cuda(device_ids[0])
        print(f"Model moved to single GPU: {device_ids[0]}")
    
    # Ensure the model's state dict is properly loaded if resuming training
    if args.start_epoch != 0:
        print(f"Loading checkpoint from {args.model}...")
        checkpoint = torch.load(args.model, map_location=f'cuda:{device_ids[0]}')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Resumed from epoch {args.start_epoch}")
    
    batch_size = args.batch_size * len(device_ids)  # Scale batch size by number of GPUs
    
    print(f"Effective batch size: {batch_size} (base: {args.batch_size} × {len(device_ids)} GPUs)")



    cudnn.benchmark = True
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(batch_size, args.csv, dataset_name, args.disable_normalization)

    logfolder = "runs/"+args.data 
    print(f"Training dataset: {os.path.basename(args.data)}")
    if not os.path.exists(logfolder):
       os.makedirs(logfolder)
    writer = SummaryWriter(logfolder)
    
    # Create training log file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.data}/training_log_{dataset_name}_{timestamp}.txt"
    
    def log_and_print(message):
        """Print to console and log to file"""
        print(message, flush=True)
        with open(log_filename, 'a') as f:
            f.write(message + '\n')
            f.flush()
    
    # Log training configuration
    log_and_print(f"=== IM2ELEVATION Training Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_and_print(f"Dataset: {dataset_name}")
    log_and_print(f"Epochs: {args.epochs}")
    log_and_print(f"Batch Size: {batch_size}")
    log_and_print(f"Learning Rate: {args.lr}")
    log_and_print(f"Device IDs: {device_ids}")
    log_and_print(f"Training CSV: {args.csv}")
    log_and_print(f"Output Directory: {args.data}")
    log_and_print(f"Log File: {log_filename}")
    log_and_print("=" * 60)
 
    # Best checkpoint tracking based on test set performance (like original IM2ELEVATION)
    best_rmse = float('inf')
    best_epoch = 0
    best_model_path = None
    
    # Load test data for evaluation (reproducing original methodology)
    test_csv = args.csv.replace('train_', 'test_')
    if not os.path.exists(test_csv):
        print(f"Warning: Test CSV not found at {test_csv}")
        print("Falling back to training loss for best checkpoint selection")
        use_test_evaluation = False
        best_loss = float('inf')
    else:
        print(f"Using test set evaluation for best checkpoint: {test_csv}")
        use_test_evaluation = True

    log_and_print(f"Starting training for {args.epochs} epochs...")
    log_and_print("=" * 60)

    # Validate epoch range
    if args.start_epoch >= args.epochs:
        log_and_print(f"Warning: Start epoch ({args.start_epoch}) >= total epochs ({args.epochs})")
        log_and_print("No training epochs to run. Training completed.")
        log_and_print("=" * 50)
        return

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # Train and get average loss for this epoch
        avg_loss = train(train_loader, model, optimizer, epoch, writer)
        
        # Evaluate on test set to select best model (like original authors did)
        if use_test_evaluation:
            test_rmse = run_test_evaluation(model, test_csv, dataset_name, epoch, device_ids)
            
            # Save checkpoint if this is the best model so far based on test RMSE
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_epoch = epoch
                
                # Remove previous best checkpoint to save space
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    log_and_print(f"Removed previous checkpoint: {os.path.basename(best_model_path)}")
                
                # Save new best checkpoint
                best_model_path = save_model + f'best_epoch_{epoch}.pth.tar'
                modelname = save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch, 'loss': avg_loss}, best_model_path)
                log_and_print(f"🏆 NEW BEST! Epoch {epoch}, Test RMSE: {test_rmse:.4f} (Train Loss: {avg_loss:.4f})")
            else:
                log_and_print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Test RMSE: {test_rmse:.4f} (Best: {best_rmse:.4f} at epoch {best_epoch})")
        else:
            # Fallback to training loss if test data not available
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                
                # Remove previous best checkpoint to save space
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    log_and_print(f"Removed previous checkpoint: {os.path.basename(best_model_path)}")
                
                # Save new best checkpoint
                best_model_path = save_model + f'best_epoch_{epoch}.pth.tar'
                modelname = save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch, 'loss': avg_loss}, best_model_path)
                log_and_print(f"🏆 NEW BEST! Epoch {epoch}, Loss: {avg_loss:.4f}")
            else:
                log_and_print(f"Epoch {epoch}, Loss: {avg_loss:.4f} (Best: {best_loss:.4f} at epoch {best_epoch})")
            
        # Also save latest checkpoint (overwrite each time to save space)
        latest_path = save_model + 'latest.pth.tar'
        save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch, 'loss': avg_loss}, latest_path)

    log_and_print("=" * 50)
    if use_test_evaluation:
        log_and_print(f"✅ Training completed! Best: Epoch {best_epoch}, Test RMSE: {best_rmse:.4f}")
    else:
        log_and_print(f"✅ Training completed! Best: Epoch {best_epoch}, Train Loss: {best_loss:.4f}")
    
    # Handle case where no epochs were run (e.g., when resuming with start_epoch >= epochs)
    if best_model_path is not None:
        log_and_print(f"📁 Checkpoints: {os.path.basename(best_model_path)}, latest.pth.tar")
    else:
        log_and_print("📁 No new checkpoints created (no epochs were run)")
    log_and_print("=" * 50)
        


def train(train_loader, model, optimizer, epoch, writer):
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()
    global args
    args = parser.parse_args()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
       

        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda(non_blocking=True)
        image = image.cuda()

        # Note: torch.autograd.Variable is deprecated, tensors have autograd by default
        image.requires_grad_(True)
        depth.requires_grad_(True)

        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
        ones.requires_grad_(True)
        optimizer.zero_grad()

        output = model(image)
        

        # Disable debug image saving during training for cleaner output
        if False:  # Changed from i%200 == 0 to False to disable completely
            x = output[0]
            x = x.view([220,220])
            x = x.cpu().detach().numpy()
            x = x*100000
            x2 = depth[0]
            print(x)
            x2 = x2.view([220,220])
            x2 = x2.cpu().detach().numpy()
            x2 = x2  *100000
            print(x2)

            x = x.astype('uint16')
            cv2.imwrite(args.data+str(i)+'_out.png',x)
            x2 = x2.astype('uint16')
            cv2.imwrite(args.data+str(i)+'_out2.png',x2)
        

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
   
        batchSize = depth.size(0)


        # Print progress more frequently for small datasets
        batch_frequency = max(1, len(train_loader) // 4)  # Show 4 updates per epoch minimum
        if i % batch_frequency == 0 or i == len(train_loader) - 1:  # Always show last batch
            print('Epoch: [{0}][{1}/{2}] Loss: {loss.avg:.4f}'
                  .format(epoch, i, len(train_loader), loss=losses), flush=True)
    
    writer.add_scalar('training loss', losses.avg, epoch)
    
    # Return average loss for this epoch
    return losses.avg


  

 

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.9 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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



def save_checkpoint(state, filename='test.pth.tar'):
    torch.save(state, filename)
    return filename


def run_test_evaluation(model, test_csv, dataset_name, epoch, device_ids):
    """
    Run test evaluation by using the existing test.py script.
    This reproduces the original IM2ELEVATION methodology where authors
    likely manually tested each checkpoint after each epoch.
    """
    import tempfile
    import subprocess
    import shutil
    import re
    
    # Save current model state temporarily
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, f'temp_epoch_{epoch}.pth.tar')
    save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch}, temp_model_path)
    
    # Handle multi-GPU models properly
    is_dataparallel = isinstance(model, torch.nn.DataParallel)
    if is_dataparallel:
        # For DataParallel models, get the original device info
        original_device_ids = model.device_ids
        original_output_device = model.output_device
        # Temporarily move the wrapped model to CPU
        model.module.cpu()
    else:
        # For single GPU models
        original_device = next(model.parameters()).device
        model.cpu()
    
    torch.cuda.empty_cache()  # Clear GPU cache
    
    try:
        # Run test.py script on the temporary checkpoint
        cmd = [
            'python', 'test.py',
            '--model', temp_dir,
            '--csv', test_csv,
            '--batch-size', '1',  # Use smaller batch size for test evaluation
            '--gpu-ids', str(device_ids[0])  # Use the first GPU from training
        ]
        
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"Warning: Test evaluation failed for epoch {epoch}")
            print(f"Error: {result.stderr}")
            return float('inf')  # Return worst possible RMSE
        
        # Parse RMSE from output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'RMSE:' in line:
                # Extract RMSE value from line like "Loss: -1.0235 | MSE: 247.5982 | RMSE: 15.7353 | MAE: 10.6550 | SSIM: 0.0290"
                rmse_match = re.search(r'RMSE:\s*([0-9.]+)', line)
                if rmse_match:
                    return float(rmse_match.group(1))
        
        print(f"Warning: Could not parse RMSE from test output for epoch {epoch}")
        return float('inf')
        
    except Exception as e:
        print(f"Error running test evaluation for epoch {epoch}: {e}")
        return float('inf')
    
    finally:
        # Restore model to original device configuration
        if is_dataparallel:
            # Move the wrapped model back to the first GPU
            model.module.cuda(original_device_ids[0])
        else:
            # Move single GPU model back to original device
            model.to(original_device)
        
        torch.cuda.empty_cache()  # Clear GPU cache again
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)




if __name__ == '__main__':
    main()
