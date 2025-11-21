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
from .util.eval import evaluate_cat, evaluate_gene, evaluate_continu
from .util.config import StepUnfreezer, WarmupCosAnnlSchedlr, setup_layer_selective_training






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
def train_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, device, gc_step=1, writer=None,
                global_step_offset=0):
    """
    Train the model for one epoch with gradient monitoring

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
    writer: tensorboard.SummaryWriter
        TensorBoard writer for logging
    global_step_offset: int
        Offset for global step (typically epoch * len(train_loader))

    Returns:
    --------
    avg_loss: float
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Initialize gradient norm tracking
    taskhead_grad_norms = []
    longnet_grad_norms = []

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
        global_step = global_step_offset + batch_idx
        embed, coords, category = batch['tile_embeds'].to(device), batch['coords'].to(device), batch['categories'].to(device)

        # Mixed precision training if scaler is provided
        if fp16_scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(embeddings=embed, coords=coords)
                loss = loss_fn(output, category)
                loss = loss / gc_step

            fp16_scaler.scale(loss).backward()

            if (batch_idx + 1) % gc_step == 0:
                # Compute gradient norms before optimizer step
                if writer is not None:
                    # Check if LongNet is frozen by checking requires_grad
                    longnet_frozen = not any(p.requires_grad for p in model.longnetmodel.parameters()) if hasattr(model,
                                                                                                                  'longnetmodel') else True

                    # Compute TaskHead gradient norm
                    if hasattr(model, 'taskhead'):
                        taskhead_grad_norm = monitor_grad_norm(model.taskhead)
                        taskhead_grad_norms.append(taskhead_grad_norm)
                        writer.add_scalar('Gradient Norm/TaskHead', taskhead_grad_norm, global_step)

                    # Compute LongNet gradient norm (only if unfrozen)
                    if hasattr(model, 'longnetmodel') and not longnet_frozen:
                        longnet_grad_norm = monitor_grad_norm(model.longnetmodel)
                        longnet_grad_norms.append(longnet_grad_norm)
                        writer.add_scalar('Gradient Norm/LongNet', longnet_grad_norm, global_step)

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
                # Compute gradient norms before optimizer step
                if writer is not None:
                    # Check if LongNet is frozen by checking requires_grad
                    longnet_frozen = not any(p.requires_grad for p in model.longnetmodel.parameters()) if hasattr(model,
                                                                                                                  'longnetmodel') else True

                    # Compute TaskHead gradient norm
                    if hasattr(model, 'taskhead'):
                        taskhead_grad_norm = monitor_grad_norm(model.taskhead)
                        taskhead_grad_norms.append(taskhead_grad_norm)
                        writer.add_scalar('Gradient Norm/TaskHead', taskhead_grad_norm, global_step)

                    # Compute LongNet gradient norm (only if unfrozen)
                    if hasattr(model, 'longnetmodel') and not longnet_frozen:
                        longnet_grad_norm = monitor_grad_norm(model.longnetmodel)
                        longnet_grad_norms.append(longnet_grad_norm)
                        writer.add_scalar('Gradient Norm/LongNet', longnet_grad_norm, global_step)

                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gc_step
        num_batches += 1

    avg_loss = total_loss / num_batches

    # Log average gradient norms for the epoch
    if writer is not None and taskhead_grad_norms:
        writer.add_scalar('Gradient Norm Avg/TaskHead', sum(taskhead_grad_norms) / len(taskhead_grad_norms), epoch)

    if writer is not None and longnet_grad_norms:
        writer.add_scalar('Gradient Norm Avg/LongNet', sum(longnet_grad_norms) / len(longnet_grad_norms), epoch)

    return avg_loss


#%%
# evaluation functions
# imported from util.eval


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
          # warm up settings
          warmup_epochs=5,
          # step unfreezer settings
          freeze_longnet=False,
          active_longnet_layers=None,  # If None, start with all frozen
          progressive_unfreeze=True,
          unfreeze_schedule=None,
          max_frozen_layers=4,  # NEW: Keep first 4 layers frozen always
          longnet_lr_factor=0.1,
          layer_lr_factor=0.2,
          min_lr_factor=0.1,
          warmup_factor=0.1
          ):
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

    # Setup progressive unfreezing if enabled
    unfreezer = None
    if progressive_unfreeze:
        if unfreeze_schedule is None:
            raise ValueError("progressive_unfreeze=True requires unfreeze_schedule to be specified")

        unfreezer = StepUnfreezer(
            model,
            initial_active_layers=active_longnet_layers,  # Can be None
            unfreeze_schedule=unfreeze_schedule,
            total_epochs=num_epochs,
            max_frozen_layers=max_frozen_layers
        )

        # Get initial trainable parameters
        trainable_longnet = unfreezer.get_trainable_params()

        param_groups = [
            {'params': model.taskhead.parameters(), 'lr': lr, 'name': 'taskhead'}
        ]

        if trainable_longnet:  # Only add if there are trainable LongNet params
            layer_base_lr = lr * longnet_lr_factor
            # Iterate in reverse order (highest layer index first)
            for layer_idx in sorted(trainable_longnet.keys(), reverse=True):
                param_groups.append({
                    'params': trainable_longnet[layer_idx],
                    'lr': layer_base_lr,
                    'name': f'longnet_layer_{layer_idx}'
                })
                layer_base_lr = layer_base_lr * layer_lr_factor

        unfreezer.print_detailed_state()

    elif freeze_longnet:
        # Freeze entire LongNet
        for param in model.longnetmodel.parameters():
            param.requires_grad = False
        param_groups = [
            {'params': model.taskhead.parameters(), 'lr': lr, 'name': 'taskhead'}
        ]
        print("Entire LongNet is frozen")

    # useless
    elif active_longnet_layers is not None:
        # Selective layer training (without progressive unfreezing)
        param_groups = setup_layer_selective_training(
            model,
            active_layers=active_longnet_layers,
            freeze_taskhead=False
        )
        for group in param_groups:
            if group['name'] == 'taskhead':
                group['lr'] = lr
            else:
                group['lr'] = lr * longnet_lr_factor

    else:
        # Train all LongNet layers
        param_groups = [
            {'params': model.taskhead.parameters(), 'lr': lr, 'name': 'taskhead'},
            {'params': model.longnetmodel.parameters(), 'lr': lr * longnet_lr_factor, 'name': 'longnet'}
        ]

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
    scheduler = WarmupCosAnnlSchedlr(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,  # Warmup for 5 epochs
        total_epochs=num_epochs,  # Total training epochs
        min_lr_factor=min_lr_factor,  # Min LR = 0.1 * base_lr = 0.01
        warmup_factor=warmup_factor  # Start warmup at 0.1 * base_lr = 0.01
    )
    print(f"Progressive warmup strategy:")
    print(f"- TaskHead warmup: {warmup_epochs} epochs (LongNet frozen)")

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
        # ========== STEP 1: CHECK FOR LAYER UNFREEZING ==========
        # This MUST happen before calculating global_step_offset
        # because we might recreate the optimizer here
        optimizer_recreated = False

        if unfreezer and unfreezer.step(epoch):
            # Recreate optimizer with newly trainable parameters
            trainable_longnet = unfreezer.get_trainable_params()

            # Recreate parameter groups
            param_groups = [
                {'params': model.taskhead.parameters(), 'lr': lr, 'name': 'taskhead'}
            ]

            if trainable_longnet:  # Only add if there are trainable LongNet params
                layer_base_lr = lr * longnet_lr_factor
                # Iterate in reverse order (highest layer index first)
                for layer_idx in sorted(trainable_longnet.keys(), reverse=True):
                    param_groups.append({
                        'params': trainable_longnet[layer_idx],
                        'lr': layer_base_lr,
                        'name': f'longnet_layer_{layer_idx}'
                    })
                    layer_base_lr = layer_base_lr * layer_lr_factor

            unfreezer.print_detailed_state()

            # Recreate optimizer
            if optim == 'sgd':
                optimizer = torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
            elif optim == 'adam':
                optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
            elif optim == 'adamw':
                optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

            # ========== STEP 2: UPDATE LEARNING RATES ==========
            # Recreate scheduler if needed
            scheduler = WarmupCosAnnlSchedlr(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,  # Warmup for 5 epochs
                total_epochs=num_epochs - epoch,  # Total training epochs
                min_lr_factor=min_lr_factor,  # Min LR = 0.1 * base_lr = 0.01
                warmup_factor=warmup_factor  # Start warmup at 0.1 * base_lr = 0.01
            )
            unfreezer.print_detailed_state()


        # ========== STEP 3: CALCULATE GLOBAL STEP OFFSET ==========
        # NOW we can calculate this (after optimizer is finalized for this epoch)
        global_step_offset = epoch * len(train_loader)

        # ========== STEP 4: TRAIN ONE EPOCH ==========
        avg_loss = train_epoch(train_loader, model, fp16_scaler, optimizer, criterion,
                               epoch, device, gc_step, writer=writer, global_step_offset=global_step_offset)

        # Get current learning rates for all parameter groups
        scheduler.step()
        current_lrs = scheduler.get_last_lr()  # This returns the current learning rates


        # ========== STEP 5: LOGGING ==========
        # Print current learning rates and frozen status
        group_names = ["TaskHead", "LongNet"]
        lr_info = ", ".join([f"{name}: {lr:.6f}" for name, lr in zip(group_names, current_lrs)])
        print(f'Epoch [{epoch + 1}/{num_epochs}]\tTrain Loss: {avg_loss:.4f}\tLRs: {lr_info}')

        # Log learning rates to TensorBoard
        writer.add_scalar('Learning Rate/TaskHead', current_lrs[0], epoch)
        if len(current_lrs) > 1:
            writer.add_scalar('Learning Rate/LongNet', current_lrs[1], epoch)

        writer.add_scalar('Train Loss', avg_loss, epoch)

        # ========== STEP 6: VALIDATION ==========
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

                # ========== STEP 7: MODEL SAVING ==========
                if metric_better(current_metric, best_metric):
                    print(f'Best F1 improved from {best_metric:.3f} to {current_metric:.3f}')
                    best_metric = current_metric
                    torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

    # ========== POST-TRAINING ==========
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
