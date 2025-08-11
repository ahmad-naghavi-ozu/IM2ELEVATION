#original script: https://github.com/fangchangma/sparse-to-dense/blob/master/utils.lua
import torch
import math
import numpy as np
import pytorch_ssim
from PIL import Image
import cv2
import torch
def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    nanMask = getNanMask(target)
   
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    #_input = torch.where(_input < torch.tensor(4), torch.tensor(0), _input)
    #_target = torch.where(_target < torch.tensor(4), torch.tensor(0), _target)


    return _input, _target, nanMask, nValidElement


def evaluateError(output, target, idx, batches, enable_clipping=False, clipping_threshold=30.0, 
                  enable_target_filtering=True, target_threshold=1.0):

    errors = {'MSE': 0, 'RMSE': 0, 'MAE': 0,'SSIM':0}
                                                                                                                                                                                                                                                                                                                                                                    
    _output, _target, nanMask, nValidElement = setNanToZero(output, target)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
   

    if (nValidElement.data.cpu().numpy() > 0):


        


        output_0_1 = _output.cpu().detach().numpy()
        target_0_1 = _target.cpu().detach().numpy()



        #x = np.reshape(output_0_1,[500,500])
        #x = x *100000
        #x = x.astype('uint16');
     
        #cv2.imwrite(str(idx)+'_out.png',x)
        
        output_0_1= output_0_1*100
        target_0_1 = target_0_1*100


        # Optional target filtering - enabled by default to maintain compatibility
        if enable_target_filtering:
            idx_zero = np.where(target_0_1 <= target_threshold)
            output_0_1[idx_zero] = 0
            if idx == 0:  # Log only once per batch to avoid spam
                filtered_count = len(idx_zero[0]) if len(idx_zero) > 0 else 0
                print(f"[FILTERING] Applied ≤{target_threshold}m target filtering, filtered {filtered_count} predictions")
        
        # Optional clipping - disabled by default to allow full height prediction range
        if enable_clipping:
            clipped_indices = np.where(output_0_1 >= clipping_threshold)
            clipped_count = len(clipped_indices[0]) if len(clipped_indices) > 0 else 0
            output_0_1[clipped_indices] = 0
            if idx == 0:  # Log only once per batch to avoid spam
                print(f"[CLIPPING] Applied ≥{clipping_threshold}m threshold, clipped {clipped_count} predictions")

       
        output_0_1 = torch.from_numpy(output_0_1).float().to(device)
        target_0_1 = torch.from_numpy(target_0_1).float().to(device)
        




        

        diffMatrix = torch.abs((output_0_1) - (target_0_1))




        IMsize = target_0_1.shape[2]*target_0_1.shape[3]

        errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / IMsize / batches
        errors['MAE'] = torch.sum(diffMatrix) / IMsize / batches
        
        # Use custom SSIM implementation to avoid pytorch_ssim padding bug
        def custom_ssim(img1, img2, window_size=11):
            """Custom SSIM implementation with integer padding"""
            import torch.nn.functional as F
            
            channel = img1.size()[1]
            window = create_window(window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            # Use integer division to avoid float padding issue
            padding = window_size // 2
            
            mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
            mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return ssim_map.mean()

        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window
        
        errors['SSIM'] = custom_ssim(_output, _target)
       
       

       
       

        errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
        errors['SSIM'] = float(errors['SSIM'].data.cpu().numpy())
        errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
        #errors['SSIM'] = float(errors['SSIM'])

    return errors


def addErrors(errorSum, errors, batchSize):
    errorSum['MSE']=errorSum['MSE'] + errors['MSE'] * batchSize
    errorSum['SSIM']=errorSum['SSIM'] + errors['SSIM'] * batchSize
    errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

    return errorSum


def averageErrors(errorSum, N):

    averageError= {'MSE': 0, 'RMSE': 0, 'MAE': 0,'SSIM':0}
    averageError['MSE'] = errorSum['MSE'] / N
    averageError['SSIM'] = errorSum['SSIM'] / N
    averageError['MAE'] = errorSum['MAE'] / N


    return averageError

def feature_plot(feats,w,h):

    feats = feats.cpu().detach().numpy()
    channel = feats.shape[1]
    feats = np.reshape(feats,(channel,w,h))
    for idx in range(16):
        m = feats[idx]*10000
        m = m.astype('uint16')
        cv2.imwrite('Fusion_features/'+str(idx)+'_mf_features.png',m)
        
    
    return feats
  






	
