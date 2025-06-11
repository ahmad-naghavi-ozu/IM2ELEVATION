import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
import cv2


class depthDataset(Dataset):
    """Dynamic input size depth dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.loc[idx, 0]
        depth_name = self.frame.loc[idx, 1]
        
        depth = cv2.imread(depth_name,-1); depth = (depth*1000).astype(np.uint16)
        depth = Image.fromarray(depth)
        image = Image.open(image_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=64, csv_data='', input_size=440):
    """
    Dynamic training data loader supporting variable input sizes.
    
    Args:
        batch_size: Batch size for training
        csv_data: Path to CSV file
        input_size: Square input size (e.g., 440, 512, 256)
    """
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    # Dynamic depth size: half of input size (following IM2ELEVATION convention)
    depth_size = input_size // 2

    csv = csv_data
    transformed_training_trans = depthDataset(csv_file=csv,
                                        transform=transforms.Compose([
                                            #RandomHorizontalFlip(),
                                            CenterCrop([input_size, input_size], [depth_size, depth_size]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training_trans, batch_size, num_workers=4, pin_memory=False)
    return dataloader_training


def getTestingData(batch_size=3, csv='', input_size=440):
    """
    Dynamic testing data loader supporting variable input sizes.
    
    Args:
        batch_size: Batch size for testing
        csv: Path to CSV file
        input_size: Square input size (e.g., 440, 512, 256)
    """
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    csvfile = csv
    transformed_testing = depthDataset(csv_file=csvfile,
                                       transform=transforms.Compose([
                                           CenterCrop([input_size, input_size], [input_size, input_size]),
                                           ToTensor(),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=12, pin_memory=False)
    return dataloader_testing


def getMultiScaleTestingData(batch_size=1, csv='', input_sizes=[256, 440, 512, 640]):
    """
    Multi-scale testing for comprehensive evaluation.
    
    Args:
        batch_size: Should be 1 for multi-scale testing
        csv: Path to CSV file
        input_sizes: List of input sizes to test
    
    Returns:
        Dictionary of dataloaders for each scale
    """
    dataloaders = {}
    for size in input_sizes:
        dataloaders[f'{size}x{size}'] = getTestingData(batch_size, csv, size)
    return dataloaders


# Utility functions for dynamic input size analysis
def get_optimal_input_size(image_path, min_size=256, max_size=1024, step=64):
    """
    Analyze image to suggest optimal input size based on content.
    """
    img = Image.open(image_path)
    w, h = img.size
    
    # Use smaller dimension as base, rounded to nearest step
    min_dim = min(w, h)
    suggested_size = ((min_dim // step) * step)
    
    # Clamp to valid range
    suggested_size = max(min_size, min(suggested_size, max_size))
    
    return suggested_size


def analyze_dataset_sizes(csv_path):
    """
    Analyze all images in dataset to find size distribution.
    """
    df = pd.read_csv(csv_path, header=None)
    sizes = []
    
    for idx in range(len(df)):
        img_path = df.iloc[idx, 0]
        try:
            img = Image.open(img_path)
            sizes.append(min(img.size))  # Use minimum dimension
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    if sizes:
        sizes = np.array(sizes)
        print(f"Dataset size analysis:")
        print(f"  Min size: {sizes.min()}")
        print(f"  Max size: {sizes.max()}")
        print(f"  Mean size: {sizes.mean():.1f}")
        print(f"  Median size: {np.median(sizes):.1f}")
        print(f"  Suggested input size: {int(np.median(sizes) // 64) * 64}")
    
    return sizes
