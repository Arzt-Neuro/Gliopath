from torch.utils.data import Dataset
from PIL import Image
from typing import List
import numpy as np
import torch
import os



class TileEncodingDataset(Dataset):
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