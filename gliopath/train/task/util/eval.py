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





def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    '''Convert the labels to one-hot encoding'''
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot


# evaluation functions
def evaluate_cat(model, criterion, val_loader, device):
    """
    Evaluate the linear probe model.

    Arguments:
    model (nn.Module): Linear probe model
    val_loader (DataLoader): DataLoader for validation set
    output_dir (str): Output directory
    """

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, category_gather = [], []
        for _, batch in enumerate(val_loader):
            embed, coords, category = batch['tile_embeds'].to(device), batch['coords'].to(device), batch[
                'categories'].to(device)
            with torch.cuda.amp.autocast():
                output = model(embeddings=embed, coords=coords)
                loss = criterion(output, category)
                total_loss += loss.item()
            # gather the predictions and categories
            pred_gather.append(output.detach().cpu().float().numpy())
            category_gather.append(category.detach().cpu().numpy())

    # calculate the accuracy, AUROC, AUPRC
    pred_gather = np.concatenate(pred_gather)
    category_gather = np.concatenate(category_gather)
    accuracy = (pred_gather.argmax(1) == category_gather).mean()
    # calculate the weighted f1 score
    f1 = f1_score(category_gather, pred_gather.argmax(1), average='weighted')
    # calculate the precision and recall, precision is not accuracy
    precision, recall, _, _ = precision_recall_fscore_support(category_gather, pred_gather.argmax(1), average='macro')

    # Convert pred_gather to float32 before using it
    pred_gather_float = pred_gather.astype(np.float32)
    auroc = roc_auc_score(to_onehot(category_gather, pred_gather.shape[1]), pred_gather_float, average='macro')
    auprc = average_precision_score(to_onehot(category_gather, pred_gather.shape[1]), pred_gather_float,
                                    average='macro')

    return accuracy, f1, precision, recall, auroc, auprc, pred_gather, category_gather

def evaluate_gene(model, criterion, val_loader, device):
    """
    Evaluate the linear probe model.

    Arguments:
    model (nn.Module): Linear probe model
    val_loader (DataLoader): DataLoader for validation set
    output_dir (str): Output directory
    """

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, category_gather = [], []
        for _, batch in enumerate(val_loader):
            embed, coords, category = batch['tile_embeds'].to(device), batch['coords'].to(device), batch[
                'categories'].to(device)
            with torch.cuda.amp.autocast():
                output = model(embeddings=embed, coords=coords)
                loss = criterion(output, category)
                total_loss += loss.item()
            # gather the predictions and categories
            pred_gather.append(output.detach().cpu().float().numpy())
            category_gather.append(category.detach().cpu().numpy())

    # calculate the accuracy, AUROC, AUPRC
    pred_gather = np.concatenate(pred_gather)
    category_gather = np.concatenate(category_gather)
    pred_binary = (pred_gather > 0.5).astype(int)  # Threshold predictions
    accuracy = (pred_binary == category_gather).mean()  # Simple element-wise accuracy
    # calculate the weighted f1 score
    f1 = f1_score(category_gather, pred_binary, average='macro', zero_division=0)
    # calculate the precision and recall, precision is not accuracy
    precision, recall, _, _ = precision_recall_fscore_support(category_gather, pred_binary, average='macro', zero_division=0)

    # Convert to float32 once
    pred_tensor = torch.tensor(pred_gather, dtype=torch.float32)
    sigmoid_preds = torch.sigmoid(pred_tensor).numpy()
    # Then use sigmoid_preds in both metrics
    auroc = roc_auc_score(category_gather, sigmoid_preds, average='macro')
    auprc = average_precision_score(category_gather, sigmoid_preds, average='macro')

    return accuracy, f1, precision, recall, auroc, auprc, pred_gather, category_gather

def evaluate_continu(model, criterion, val_loader, device):
    """
    Evaluate the linear probe model.

    Arguments:
    model (nn.Module): Linear probe model
    val_loader (DataLoader): DataLoader for validation set
    output_dir (str): Output directory
    """

    model.eval()
    with torch.no_grad():
        total_loss = 0
        pred_gather, target_gather = [], []

        for _, batch in enumerate(val_loader):
            embed, coords, target = batch['tile_embeds'].to(device), batch['coords'].to(device), batch[
                'categories'].to(device)
            with torch.cuda.amp.autocast():
                output = model(embeddings=embed, coords=coords)  # DON'T squeeze for multiple outputs
                loss = criterion(output, target)
                total_loss += loss.item()

            pred_gather.append(output.detach().cpu().float().numpy())
            target_gather.append(target.detach().cpu().numpy())

    # After concatenation, ensure float32 precision
    pred_gather = np.concatenate(pred_gather, axis=0).astype(np.float32)
    target_gather = np.concatenate(target_gather, axis=0).astype(np.float32)

    # Calculate metrics across ALL variables (averaged)
    mae = np.mean(np.abs(pred_gather - target_gather))
    rmse = np.sqrt(np.mean((pred_gather - target_gather) ** 2))

    ss_res = np.sum((target_gather - pred_gather) ** 2)
    ss_tot = np.sum((target_gather - np.mean(target_gather, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return mae, rmse, r2, 0, 0, 0, pred_gather, target_gather