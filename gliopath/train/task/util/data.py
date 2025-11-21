import torch.nn as nn
import os
import io
import argparse
import zipfile
import pandas as pd

import torch
import itertools
import numpy as np
from torch import nn
from sklearn import metrics
import math

from tqdm.notebook import tqdm
from glob import glob
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import torch.utils.tensorboard as tensorboard
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import _LRScheduler




class EmbedMatcher:
    def __init__(self, sample_ids, embed_path):
        self.sample_ids = sample_ids
        self.embed_path = embed_path

    def load_embeds(self):
        if self.embed_path.endswith('.pt'):
            collated_dict = torch.load(self.embed_path)
            # Extract tile_embeds from each sample's dictionary
            embed_dict = {}
            for key in self.sample_ids:
                sample_data = collated_dict[key]
                if isinstance(sample_data, dict):
                    embed_dict[key] = {
                        'tile_embeds': sample_data['tile_embeds'],
                        'coords': sample_data.get('coords', None)  # coords might be optional
                    }
                else:
                    # Fallback if old format (direct tensor)
                    embed_dict[key] = {'tile_embeds': sample_data, 'coords': None}

        else:
            if len(glob(self.embed_path + '*.pt')) < 2:
                raise ImportError(
                    'Need either tensor dict or respective sample pt tensor files! For insufficient tensor files under the appointed directory.')

            embed_dict = {}
            for sample_id in self.sample_ids:
                path = self.embed_path + '/' + sample_id + '.pt'
                path = path.replace('//', '/')
                sample_data = torch.load(path)

                # Handle dictionary format
                if isinstance(sample_data, dict):
                    embed_dict[sample_id] = {
                        'tile_embeds': sample_data['tile_embeds'],
                        'coords': sample_data.get('coords', None)
                    }
                    if embed_dict[sample_id]['coords'] is None:
                        raise ValueError('Saved sample tensor must be dictionary with keys, tile_embeds and coords.')
                else:
                    # Fallback if old format (direct tensor)
                    raise ValueError('Saved sample tensor must be dictionary with keys, tile_embeds and coords.')

        return embed_dict

class EmbeddingDataset(Dataset):
    def __init__(self, dataset_csv:str, embed_path:str, feat_layer:list=[-1], split_col:str='split_col', split:str='train', id_col:str='id', type_col='tumour_type', outcome_type=None, z_score=False):
        """
        Dataset used for training the linear probe based on the embeddings extracted from the pre-trained model.

        Arguments:
        dataset_csv (str): Path to the csv file containing the embeddings and labels.
        """
        if type(dataset_csv) == str:
            df = pd.read_csv(dataset_csv)
        elif type(dataset_csv) == pd.DataFrame:
            df = dataset_csv
        else:
            raise TypeError('dataset_csv must be either a string (file path) or a pandas dataframe')
        split_df = df[df[split_col] == split]

        self.outcome_type = outcome_type
        if self.outcome_type is None:
            raise ValueError('Must provide type of outcome (any one of: cat, gene, continu)!')

        self.samples = split_df[id_col].tolist()
        if outcome_type == 'cat':
            self.labels = split_df[type_col].tolist()
            # generate a dict for labels
            label_set = list(self.labels)
            label_set = sorted(set(label_set))
            self.label_dict = {label: i for i, label in enumerate(label_set)}
        else:
            self.labels = split_df[type_col].values.tolist()

        # load the embeddings
        self.matcher = EmbedMatcher(self.samples, embed_path)
        self.embeds = self.matcher.load_embeds()
        self.feat_layer = feat_layer

        # if need to convert to z-score
        self.z_score = z_score

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, category = self.samples[index], self.labels[index]

        # Get the sample dictionary
        sample_dict = self.embeds[sample_id]

        # Extract tile embeddings
        embed = sample_dict['tile_embeds']  # Shape: [num_tiles, embed_dim]
        coords = sample_dict['coords']  # Shape: [num_tiles, 2]

        # need to add code for dealing batch size = 1

        if self.z_score:
            # z-score normalization
            embed = (embed - embed.mean()) / embed.std()

        # convert the label to index
        if self.outcome_type == 'cat':
            category = self.label_dict[category]
        else:
            category = torch.tensor(category, dtype=torch.float32)

        # Return dictionary format with coords (useful for spatial models)
        return {
            'tile_embeds': embed,  # [num_tiles, embed_dim]
            'coords': coords,  # [num_tiles, 2] or None
            'category': category
        }


def collate_fn_with_padding(batch):
    """
    Custom collate function to handle variable-length embeddings
    Now handles dictionary format with tile_embeds and coords

    Arguments:
    ----------
    batch: list of dicts
        Each dict contains:
        - 'embed': [L_i, D] tensor
        - 'coords': [L_i, 2] tensor or None
        - 'category': scalar or tensor

    Returns:
    --------
    dict with padded embeddings, coords, masks, and labels
    """
    # Extract components
    embeds = [item['tile_embeds'] for item in batch]
    coords_list = [item['coords'] for item in batch]
    categories = [item['category'] for item in batch]

    # Find maximum length in this batch
    max_len = max([e.size(0) for e in embeds])
    embed_dim = embeds[0].size(-1)

    # Pad embeddings
    padded_embeds = []
    padded_coords = []
    masks = []

    for i, embed in enumerate(embeds):
        seq_len = embed.size(0)

        # Pad embeddings
        padded = torch.zeros(max_len, embed_dim)
        padded[:seq_len] = embed
        padded_embeds.append(padded)

        # Pad coordinates if they exist
        if coords_list[i] is not None:
            coord = coords_list[i]
            padded_coord = torch.zeros(max_len, 2)
            padded_coord[:seq_len] = coord
            padded_coords.append(padded_coord)
        else:
            # Create dummy coords if not provided
            padded_coords.append(torch.zeros(max_len, 2))

        # Create mask (1 for real data, 0 for padding)
        mask = torch.zeros(max_len)
        mask[:seq_len] = 1
        masks.append(mask)

    # Stack into batches
    padded_embeds = torch.stack(padded_embeds)  # [B, max_len, D]
    padded_coords = torch.stack(padded_coords)  # [B, max_len, 2]
    masks = torch.stack(masks).bool()  # [B, max_len]

    # Handle categories
    categories = torch.stack(categories)

    return {
        'tile_embeds': padded_embeds,
        'coords': padded_coords,
        'masks': masks,
        'categories': categories
    }