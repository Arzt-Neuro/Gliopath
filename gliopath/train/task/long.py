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
from util.eval import evaluate_cat, evaluate_gene, evaluate_continu





#%%


#%%
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

def monitor_grad_norm(module):
    """
    Compute gradient norm for a specific module

    Parameters:
    -----------
    module: nn.Module
        Module to compute gradient norm for

    Returns:
    --------
    float: Gradient norm
    """
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


#%%
# plot the learning rate warm up curve
def plot_warmup(lr=0.001, longnet_lr_factor=0.1, taskhead_warmup=10, longnet_warmup=15, total_epochs=100,
                            min_lr=1e-7, save_dir=None):
    """
    Plot the learning rate and freezing schedule for the progressive warmup strategy
    """
    import matplotlib.pyplot as plt
    import numpy as np

    taskhead_base_lr = lr
    longnet_base_lr = lr * longnet_lr_factor

    # Generate learning rate schedules
    epochs = np.arange(total_epochs)

    # TaskHead learning rate
    taskhead_lrs = []
    for epoch in epochs:
        if epoch < taskhead_warmup:
            # Warmup phase
            taskhead_lr = taskhead_base_lr * (epoch + 1) / taskhead_warmup
        else:
            # Cosine annealing phase
            progress = (epoch - taskhead_warmup) / max(1, total_epochs - taskhead_warmup)
            taskhead_lr = min_lr + 0.5 * (taskhead_base_lr - min_lr) * (1 + np.cos(np.pi * progress))

        taskhead_lrs.append(taskhead_lr)

    # LongNet learning rate
    longnet_lrs = []
    for epoch in epochs:
        if epoch < taskhead_warmup:
            # Frozen during TaskHead warmup
            longnet_lr = 0.0
        else:
            # Calculate epochs from LongNet's perspective
            longnet_epoch = epoch - taskhead_warmup

            if longnet_epoch < longnet_warmup:
                # LongNet warmup phase
                longnet_lr = longnet_base_lr * (longnet_epoch + 1) / longnet_warmup
            else:
                # Cosine annealing phase
                remaining_epochs = total_epochs - taskhead_warmup
                progress = (longnet_epoch - longnet_warmup) / max(1, remaining_epochs - longnet_warmup)
                longnet_lr = min_lr + 0.5 * (longnet_base_lr - min_lr) * (1 + np.cos(np.pi * progress))

        longnet_lrs.append(longnet_lr)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Learning rates
    ax1 = plt.subplot(211)
    ax1.plot(epochs, taskhead_lrs, 'b-', label='TaskHead LR')
    ax1.plot(epochs, longnet_lrs, 'r-', label='LongNet LR')

    ax1.axvline(x=taskhead_warmup, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(x=taskhead_warmup + longnet_warmup, color='gray', linestyle='--', alpha=0.7)

    ax1.set_xlim(0, total_epochs)
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Progressive Warmup Strategy')
    ax1.legend()
    ax1.grid(True)

    # Freezing status
    ax2 = plt.subplot(212, sharex=ax1)
    frozen_status = [1 if e < taskhead_warmup else 0 for e in epochs]
    ax2.step(epochs, frozen_status, 'g-', where='post', label='LongNet Frozen')

    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Unfrozen', 'Frozen'])
    ax2.set_xlabel('Epoch')
    ax2.set_title('LongNet Freezing Status')
    ax2.grid(True)

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)
    else:
        save_dir = 'Plot file not saved.'
    plt.close()

    return save_dir


#%%
def train_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, device, gc_step=1):
    """
    Train the model for one epoch.

    Arguments:
    ----------
    train_loader: DataLoader
        DataLoader for training set
    model: nn.Module
        The model to train
    fp16_scaler: torch.cuda.amp.GradScaler
        Scaler for mixed precision training (can be None if not using fp16)
    optimizer: torch.optim.Optimizer
        Optimizer for training
    loss_fn: nn.Module
        Loss function
    epoch: int
        Current epoch number
    device: torch.device
        Device to train on
    gc_step: int
        Gradient accumulation steps

    Returns:
    --------
    avg_loss: float
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
        embed, coords, category = batch['tile_embeds'].to(device), batch['coords'].to(device), batch['categories'].to(device)

        # Mixed precision training if scaler is provided
        if fp16_scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(embeddings=embed, coords=coords)
                loss = loss_fn(output, category)
                loss = loss / gc_step

            fp16_scaler.scale(loss).backward()

            if (batch_idx + 1) % gc_step == 0:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                optimizer.zero_grad()
        else:
            # Regular training
            with torch.cuda.amp.autocast():
                output = model(embeddings=embed, coords=coords)
                loss = loss_fn(output, category)
                loss = loss / gc_step
            loss.backward()

            if (batch_idx + 1) % gc_step == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gc_step
        num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss


#%%
# evaluation functions

#%%
def train(model,
          train_loader,
          val_loader,
          test_loader,
          num_epochs=100,
          lr=0.01,
          min_lr=0.0,
          optim='sgd',
          weight_decay=0.0,
          output_dir='output',
          eval_interval=1,  # Now in epochs, not iterations
          momentum=0.0,
          model_select='best',
          outcome_type=None,
          gc_step=1,
          use_fp16=False,
          freeze_longnet=True,
          longnet_lr_factor=0.1):
    """
    Train the linear probe model using epoch-based training.

    Arguments:
    ----------
    model: nn.Module
        Linear probe model
    train_loader: DataLoader
        DataLoader for training set
    val_loader: DataLoader
        DataLoader for validation set
    test_loader: DataLoader
        DataLoader for test set
    num_epochs: int
        Number of training epochs
    lr: float
        Learning rate
    min_lr: float
        Minimum learning rate
    optim: str
        Optimizer
    weight_decay: float
        Weight decay
    output_dir: str
        Output directory
    eval_interval: int
        Evaluation interval (in epochs)
    momentum: float
        Momentum
    model_select: 'best' or 'last'
        Choose either the best model or the last
    outcome_type: str
        Data type of the outcome: 'cat', 'bin', 'gene', 'continu'
    gc_step: int
        Gradient accumulation steps
    use_fp16: bool
        Whether to use mixed precision training
    """
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)

    # Freeze the LongNet if requested
    if freeze_longnet:
        for param in model.longnetmodel.parameters():
            param.requires_grad = False
        param_groups = [
            {'params': list(model.taskhead.parameters()), 'lr': lr}
        ]
        print("LongNet encoder has been frozen")
    else:
        # Set up parameter groups with different learning rates
        longnet_params = list(model.longnetmodel.parameters())
        taskhead_params = list(model.taskhead.parameters())
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': taskhead_params, 'lr': lr},
            {'params': longnet_params, 'lr': lr * longnet_lr_factor}
        ]
        print(f"LongNet learning rate set to {lr * longnet_lr_factor} (TaskHead: {lr})")

    # Set the optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    print(f'Set the optimizer as {optim}')

    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    # Set up mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # Set the loss function and evaluator based on outcome type
    if outcome_type == 'cat':
        criterion = nn.CrossEntropyLoss()
        evaluate = evaluate_cat
        best_metric = 0
        metric_name = 'f1'
        metric_better = lambda new, old: new > old
    elif outcome_type == 'gene':
        criterion = nn.BCEWithLogitsLoss()
        evaluate = evaluate_gene
        best_metric = 0
        metric_name = 'f1'
        metric_better = lambda new, old: new > old
    elif outcome_type == 'bin':
        criterion = nn.CrossEntropyLoss()
        evaluate = evaluate_cat
        best_metric = 0
        metric_name = 'f1'
        metric_better = lambda new, old: new > old
    elif outcome_type == 'continu':
        criterion = nn.MSELoss()
        evaluate = evaluate_continu
        best_metric = float('inf')
        metric_name = 'mae'
        metric_better = lambda new, old: new < old
    else:
        raise ValueError('Must provide type of outcome (any one of: cat, gene, bin, continu)!')

    # Training loop
    print(f'Start training for {num_epochs} epochs with gradient accumulation steps: {gc_step}')

    for epoch in tqdm(range(num_epochs)):
        # Train one epoch
        avg_loss = train_epoch(train_loader, model, fp16_scaler, optimizer, criterion,
                               epoch, device, gc_step)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}]\tTrain Loss: {avg_loss:.4f}\tLR: {current_lr:.6f}')

        writer.add_scalar('Train Loss', avg_loss, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # Step the learning rate scheduler
        lr_scheduler.step()

        # Evaluate at specified intervals
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == num_epochs:
            print(f'Evaluating at epoch {epoch + 1}...')

            if outcome_type == 'continu':
                mae, rmse, r2, _, _, _, pred_gather, target_gather = evaluate(model, criterion, val_loader, device)
                print(f'Val Epoch [{epoch + 1}/{num_epochs}] MAE: {mae:.3f} RMSE: {rmse:.3f} R²: {r2:.3f}')

                writer.add_scalar('Val MAE', mae, epoch)
                writer.add_scalar('Val RMSE', rmse, epoch)
                writer.add_scalar('Val R²', r2, epoch)
                writer.add_scalar('Best MAE', best_metric, epoch)

                current_metric = mae

                if metric_better(current_metric, best_metric):
                    print(f'Best MAE improved from {best_metric:.3f} to {current_metric:.3f}')
                    best_metric = current_metric
                    torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

            else:
                accuracy, f1, precision, recall, auroc, auprc, pred_gather, target_gather = evaluate(
                    model, criterion, val_loader, device)
                print(f'Val Epoch [{epoch + 1}/{num_epochs}] Acc: {accuracy:.3f} F1: {f1:.3f} Prec: {precision:.3f} Rec: {recall:.3f} AUROC: {auroc:.3f} AUPRC: {auprc:.3f}')

                writer.add_scalar('Val Accuracy', accuracy, epoch)
                writer.add_scalar('Val F1', f1, epoch)
                writer.add_scalar('Val AUROC', auroc, epoch)
                writer.add_scalar('Val AUPRC', auprc, epoch)
                writer.add_scalar('Val Precision', precision, epoch)
                writer.add_scalar('Val Recall', recall, epoch)
                writer.add_scalar('Best F1', best_metric, epoch)

                current_metric = f1

                if metric_better(current_metric, best_metric):
                    print(f'Best F1 improved from {best_metric:.3f} to {current_metric:.3f}')
                    best_metric = current_metric
                    torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

    # Save the final model
    torch.save(model.state_dict(), f'{output_dir}/last_model.pth')

    # Load the selected model for final evaluation
    if model_select == 'best':
        print(f'Loading best model with {metric_name}: {best_metric:.3f}')
        model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
        val_metric = best_metric
    else:
        print('Using last model')
        model.load_state_dict(torch.load(f'{output_dir}/last_model.pth'))
        val_metric = current_metric

    # Final test evaluation
    print('Evaluating on test set...')
    if outcome_type == 'continu':
        mae, rmse, r2, _, _, _, pred_gather, target_gather = evaluate(model, criterion, test_loader, device)
        print(f'Test MAE: {mae:.3f} RMSE: {rmse:.3f} R²: {r2:.3f}')

        writer.add_scalar('Test MAE', mae, num_epochs)
        writer.add_scalar('Test RMSE', rmse, num_epochs)
        writer.add_scalar('Test R²', r2, num_epochs)

        with open(f'{output_dir}/results.txt', 'w') as f:
            f.write(f'Val MAE: {val_metric:.3f}\n')
            f.write(f'Test MAE: {mae:.3f} Test RMSE: {rmse:.3f} Test R²: {r2:.3f}\n')

    else:
        accuracy, f1, precision, recall, auroc, auprc, pred_gather, target_gather = evaluate(model, criterion,
                                                                                             test_loader, device)
        print(
            f'Test Acc: {accuracy:.3f} F1: {f1:.3f} Prec: {precision:.3f} Rec: {recall:.3f} AUROC: {auroc:.3f} AUPRC: {auprc:.3f}')

        writer.add_scalar('Test Accuracy', accuracy, num_epochs)
        writer.add_scalar('Test F1', f1, num_epochs)
        writer.add_scalar('Test AUROC', auroc, num_epochs)
        writer.add_scalar('Test AUPRC', auprc, num_epochs)
        writer.add_scalar('Test Precision', precision, num_epochs)
        writer.add_scalar('Test Recall', recall, num_epochs)

        with open(f'{output_dir}/results.txt', 'w') as f:
            f.write(f'Val F1: {val_metric:.3f}\n')
            f.write(f'Test Acc: {accuracy:.3f} F1: {f1:.3f} AUROC: {auroc:.3f} AUPRC: {auprc:.3f}\n')

    writer.close()

    return pred_gather, target_gather
