import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
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

parser.add_argument('--data', default='adjust')
parser.add_argument('--csv', default='')
parser.add_argument('--model', default='')

args = parser.parse_args()
# Extract dataset name from path for model naming
dataset_name = os.path.basename(args.data.rstrip('/'))
save_model = args.data+'/'+dataset_name+'_model_'
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
    args = parser.parse_args()
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
 
    
    if args.start_epoch != 0:
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
        model = model.cuda()
        state_dict = torch.load(args.model)['state_dict']
        model.load_state_dict(state_dict)
        batch_size = 2
    else:
        model = model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
        batch_size = 2



    cudnn.benchmark = True
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(batch_size,args.csv)

    logfolder = "runs/"+args.data 
    print(f"Training dataset: {os.path.basename(args.data)}")
    if not os.path.exists(logfolder):
       os.makedirs(logfolder)
    writer = SummaryWriter(logfolder)
 
    # Best checkpoint tracking
    best_loss = float('inf')
    best_epoch = 0
    best_model_path = None

    print(f"Starting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # Train and get average loss for this epoch
        avg_loss = train(train_loader, model, optimizer, epoch, writer)
        
        # Only save checkpoint if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            
            # Remove previous best checkpoint to save space
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Removed previous checkpoint: {os.path.basename(best_model_path)}", flush=True)
            
            # Save new best checkpoint
            best_model_path = save_model + f'best_epoch_{epoch}.pth.tar'
            modelname = save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch, 'loss': avg_loss}, best_model_path)
            print(f"🏆 NEW BEST! Epoch {epoch}, Loss: {avg_loss:.4f}", flush=True)
        else:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f} (Best: {best_loss:.4f} at epoch {best_epoch})", flush=True)
            
        # Also save latest checkpoint (overwrite each time to save space)
        latest_path = save_model + 'latest.pth.tar'
        save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch, 'loss': avg_loss}, latest_path)

    print("=" * 50, flush=True)
    print(f"✅ Training completed! Best: Epoch {best_epoch}, Loss: {best_loss:.4f}", flush=True)
    print(f"📁 Checkpoints: {os.path.basename(best_model_path)}, latest.pth.tar", flush=True)
    print("=" * 50, flush=True)
        


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




if __name__ == '__main__':
    main()
