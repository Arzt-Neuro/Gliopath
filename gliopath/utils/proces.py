import torch
import numpy as np
import os
from torchvision import transforms
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm


# set random seed to ensure replicablility
def seed_torch(device, seed=7):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# transform to one-hot format
def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    '''Convert the labels to one-hot encoding'''
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot

#%%
# load transform algorithm either simple or complicated (robust) one
def load_transforms(img_size=224, strong_augment=True):
    """
    Load transforms optimized for brain tumor pathology images

    Args:
        img_size: Target image size
        strong_augment: Whether to apply strong augmentations for better generalization
    """
    if strong_augment:
        # Strong augmentations for better transfer learning
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),  # Brain scans can be rotated
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # Add noise for robustness
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        ])
    else:
        # Standard preprocessing similar to original GigaPath
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    return transform

#%%
# split as list
def split_dataset(df: pd.DataFrame, id_col='id', type_col='tumour_type', val_split=0.2, test_split=0.1, in_df=False, split_col='split_col', random_state=42):
    """
    df: the metadata dataframe mapping patient ids, whole slide file addresses, tumor type or so
    id_col: the column name of patient IDs
    type_col: the column name of tumour type
    val_split: the proportion of validation data
    test_split: the proportion of test data

    :return: a dict containing three elements. Keys are: 'train', 'val', 'test', representing training dataset, validation dataset and test dataset. Each key retrieves a list of patient IDs, with which you can get the file addresses of corresponding data.
    """

    if val_split < 0 or test_split < 0:
        raise ValueError('Split proportions must be non-negative')
    if val_split + test_split >= 1:
        raise ValueError('val_split + test_split must be less than 1')

    # Shuffle the DataFrame rows at the start
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if int(df[id_col].duplicated().sum()) > 0:
        raise ValueError('Duplicated patient IDs!')
    if len(df[id_col]) != len(df[type_col]):  # FIXED
        raise ValueError('Numbers of patient IDs and tumour types do not match!')

    tumour_types = df[type_col].unique()
    train = []
    val = []
    test = []

    if test_split == 0:
        for tumour_type in tumour_types:
            tumour_ids = df[df[type_col] == tumour_type][id_col].tolist()
            # Split into train/val/test
            train_ids, val_ids = train_test_split(
                tumour_ids, test_size=val_split, random_state=random_state, shuffle=True
            )

            train.extend(train_ids)
            val.extend(val_ids)

        if in_df:
            df.loc[df[id_col].isin(train), split_col] = 'train'
            df.loc[df[id_col].isin(val), split_col] = 'val'

            return df

        else:
            return train, val
    else:
        for tumour_type in tumour_types:
            tumour_ids = df[df[type_col] == tumour_type][id_col].tolist()
            # Split into train/val/test
            train_ids, val_n_test = train_test_split(
                tumour_ids, test_size=(val_split + test_split), random_state=random_state, shuffle=True
            )
            val_ids, test_ids = train_test_split(
                val_n_test, test_size=test_split / (val_split + test_split), random_state=random_state, shuffle=True
            )

            train.extend(train_ids)
            val.extend(val_ids)
            test.extend(test_ids)

        if in_df:
            df.loc[df[id_col].isin(train), split_col] = 'train'
            df.loc[df[id_col].isin(val), split_col] = 'val'
            df.loc[df[id_col].isin(test), split_col] = 'test'

            return df

        else:
            return train, val, test


#%%
# split and copy
def organize_with_metadata(df, source_path, output_path, val_split, test_split):
    """Organize data using metadata file"""

    # Group by tumor type
    for tumor_type in df['tumor_type'].unique():
        tumor_files = df[df['tumor_type'] == tumor_type]['file_path'].tolist()

        # Split into train/val/test
        train_files, temp_files = train_test_split(
            tumor_files, test_size=(val_split + test_split), random_state=42, shuffle=True
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=test_split / (val_split + test_split), random_state=42, shuffle=True
        )

        # Copy files to appropriate directories
        copy_files(train_files, source_path, output_path / 'train' / tumor_type)
        copy_files(val_files, source_path, output_path / 'val' / tumor_type)
        copy_files(test_files, source_path, output_path / 'test' / tumor_type)

        print(f"{tumor_type}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


def organize_by_directory_structure(source_path, output_path, val_split, test_split):
    """Organize data based on existing directory structure"""

    tumor_types = ['normal', 'glioma', 'meningioma', 'pituitary']

    for tumor_type in tumor_types:
        tumor_dir = source_path / tumor_type
        if not tumor_dir.exists():
            print(f"Warning: {tumor_dir} does not exist, skipping...")
            continue

        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        all_files = []
        for ext in image_extensions:
            all_files.extend(list(tumor_dir.glob(f'*{ext}')))
            all_files.extend(list(tumor_dir.glob(f'*{ext.upper()}')))

        if not all_files:
            print(f"No images found in {tumor_dir}")
            continue

        # Split files
        train_files, temp_files = train_test_split(
            all_files, test_size=(val_split + test_split), random_state=42, shuffle=True
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=test_split / (val_split + test_split), random_state=42, shuffle=True
        )

        # Copy files
        copy_files(train_files, Path(), output_path / 'train' / tumor_type, absolute_paths=True)
        copy_files(val_files, Path(), output_path / 'val' / tumor_type, absolute_paths=True)
        copy_files(test_files, Path(), output_path / 'test' / tumor_type, absolute_paths=True)

        print(f"{tumor_type}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


def copy_files(file_list, source_base, target_dir, absolute_paths=False):
    """Copy files from source to target directory"""
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(file_list, desc=f"Copying to {target_dir.name}"):
        if absolute_paths:
            source_file = file_path
        else:
            source_file = source_base / file_path

        target_file = target_dir / source_file.name
        shutil.copy2(source_file, target_file)


def create_metadata_template(output_file):
    """Create a template metadata CSV file"""

    # Example metadata structure
    sample_data = {
        'file_path': [
            'patient_001_tile_001.png',
            'patient_001_tile_002.png',
            'patient_002_tile_001.png',
            'patient_003_tile_001.png'
        ],
        'patient_id': ['patient_001', 'patient_001', 'patient_002', 'patient_003'],
        'tumor_type': ['glioma', 'glioma', 'normal', 'meningioma'],
        'grade': [2, 2, 0, 1],  # Tumor grade (0 for normal)
        'coordinates_x': [256, 512, 256, 768],
        'coordinates_y': [256, 256, 512, 256],
        'magnification': [20, 20, 20, 20]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv(output_file, index=False)
    print(f"Metadata template created at: {output_file}")
    print("Please fill in your actual data information.")

#%%
# calculate the accuracy of the model
