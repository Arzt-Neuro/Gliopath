from torch.utils.data import DataLoader, Dataset
from monai.data import Dataset as m_Data
from monai.data import DataLoader as m_Loader
import pandas as pd
import numpy as np
import torch
import os

from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import List, Tuple, Union




# used in first step model
class TileDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
        Example: ['images/256x_256y.png', 'images/256x_512y.png']
    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """
    def __init__(self, image_paths: List[str], transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split('.png')[0].split('_')
        x, y = int(x.replace('x', '')), int(y.replace('y', ''))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {'img': torch.from_numpy(np.array(img)),
                'coords': torch.from_numpy(np.array([x, y])).float()}


# unfinished class for fine-tuning 2nd step model
class SlideDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
        Example: ['images/256x_256y.png', 'images/256x_512y.png']
    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """
    def __init__(self, image_paths: List[str], transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split('.png')[0].split('_')
        x, y = int(x.replace('x', '')), int(y.replace('y', ''))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {'img': torch.from_numpy(np.array(img)),
                'coords': torch.from_numpy(np.array([x, y])).float()}


# weighted sampler
def get_sampler(train_dataset):
    from torch.utils.data import WeightedRandomSampler
    # get the weights for each class, we only do this for multi-class classification
    N = len(train_dataset)
    weights = {}
    for idx in range(N):
        label = int(train_dataset.labels[idx][0])
        if label not in weights: weights[label] = 0
        weights[label] += 1.0 / N
    for l in weights.keys(): weights[l] = 1.0 / weights[l]
    sample_weights = [weights[int(train_dataset.labels[i][0])] for i in range(N)]
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return train_sampler

# optimizer


# scheduler


# loss function


# accuracy function: accuracy, F1 score, AUC



