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




class SeriesHead(nn.Module):

    def __init__(self, LongNetModel:nn.Module=None, TaskHead:nn.Module=None, feat_layers:list=[-1], z_score:bool=False):
        super(SeriesHead, self).__init__()

        self.longnetmodel = LongNetModel
        self.taskhead = TaskHead

        self.feat_layer = feat_layers
        self.z_score = z_score

    def forward(self, embeddings, coords):
        """
        Forward pass with optional coordinates and masking

        Arguments:
        ----------
        embeddings: torch.Tensor [B, L, D] or [L, D]
            Input tile embeddings (possibly padded)
        coords: torch.Tensor [B, L, 2] or [L, 2], optional
            Spatial coordinates of tiles
        mask: torch.Tensor [B, L] or [L], optional
            Boolean mask (True for real data, False for padding)
        """
        # Pass embeddings and coords through LongNet
        embed = self.longnetmodel.forward(embeddings, coords, all_layer_embed=True)

        # Extract features from specified layers
        embed = [embed[i] for i in self.feat_layer]
        embed = torch.cat(embed, dim=-1)

        if self.z_score:
            embed = (embed - embed.mean()) / embed.std()

        # Pass through task head
        embed = self.taskhead(embed)
        return embed


#%%



#%%
# the model head
class TaskHead(nn.Module):

    def __init__(self, embed_dim:int=1536, num_classes:int=10, lite:bool=True, long:bool=False, dropout:float=0.1, middle_dim:int=1024, hidden_dim:int=512):
        super(TaskHead, self).__init__()

        if lite == long:
            raise ValueError('Cannot be both lite and long!')

        if long:
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, middle_dim),
                nn.BatchNorm1d(middle_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(middle_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )

        elif lite:
            self.fc = nn.Linear(embed_dim, num_classes)

        else:
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, num_classes)
            )

    def forward(self, embeddings):
        return self.fc(embeddings)


class CosineAnnealingWarmupScheduler:
    """
    Simple scheduler with warmup and cosine annealing
    Works with ProgressiveLayerUnfreezer for layer control
    Does NOT handle freezing (let ProgressiveLayerUnfreezer do that)
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._step_count = 0

    def get_lr(self, base_lr, epoch):
        """Calculate learning rate with warmup + cosine annealing"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, epoch=None):
        """Update learning rates for all parameter groups"""
        if epoch is None:
            epoch = self._step_count
            self._step_count += 1

        # Update LR for all parameter groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.get_lr(self.base_lrs[i], epoch)

        return [group['lr'] for group in self.optimizer.param_groups]


class SchedulerWarmup:
    """
    Scheduler that:
    1. Freezes LongNet during TaskHead warmup
    2. Unfreezes LongNet after TaskHead warmup is complete
    3. Applies separate warmup for LongNet after unfreezing
    """

    def __init__(self,
                 optimizer,
                 model,
                 taskhead_warmup_epochs,
                 longnet_warmup_epochs,
                 total_epochs,
                 min_lr=1e-7,
                 last_epoch=-1):
        self.optimizer = optimizer
        self.model = model
        self.taskhead_warmup_epochs = taskhead_warmup_epochs
        self.longnet_warmup_epochs = longnet_warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._step_count = 0

        # Freeze LongNet initially
        self._freeze_longnet(True)
        self.longnet_frozen = True

        # Initialize learning rates to starting values
        self.step()

    def _freeze_longnet(self, freeze=True):
        """Freeze or unfreeze LongNet model parameters"""
        if hasattr(self.model, 'longnetmodel'):
            for param in self.model.longnetmodel.parameters():
                param.requires_grad = not freeze
            self.longnet_frozen = freeze
            status = "frozen" if freeze else "unfrozen"
            print(f"LongNet is now {status}")

    def get_taskhead_lr(self, base_lr, epoch):
        """Calculate learning rate for TaskHead"""
        if epoch < self.taskhead_warmup_epochs:
            # Linear warmup phase
            return base_lr * (epoch + 1) / self.taskhead_warmup_epochs
        else:
            # Cosine annealing phase after warmup
            progress = (epoch - self.taskhead_warmup_epochs) / max(1, self.total_epochs - self.taskhead_warmup_epochs)
            return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_longnet_lr(self, base_lr, epoch):
        """
        Calculate learning rate for LongNet
        - LR=0 during TaskHead warmup (while frozen)
        - Linear warmup after unfreezing
        - Cosine annealing after LongNet warmup
        """
        # Calculate the actual epoch from LongNet's perspective
        longnet_epoch = epoch - self.taskhead_warmup_epochs

        if epoch < self.taskhead_warmup_epochs:
            # During TaskHead warmup, LongNet is frozen with LR=0
            return 0.0
        elif longnet_epoch < self.longnet_warmup_epochs:
            # LongNet warmup phase after unfreezing
            return base_lr * (longnet_epoch + 1) / self.longnet_warmup_epochs
        else:
            # Cosine annealing phase after warmup
            remaining_epochs = self.total_epochs - self.taskhead_warmup_epochs
            progress = (longnet_epoch - self.longnet_warmup_epochs) / max(1,
                                                                          remaining_epochs - self.longnet_warmup_epochs)
            return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, epoch=None):
        if epoch is None:
            epoch = self._step_count
            self._step_count += 1

        # Check if we need to unfreeze LongNet at this epoch
        if self.longnet_frozen and epoch >= self.taskhead_warmup_epochs:
            self._freeze_longnet(False)  # Unfreeze LongNet

        # Update learning rate for TaskHead
        self.optimizer.param_groups[0]['lr'] = self.get_taskhead_lr(self.base_lrs[0], epoch)

        # Update learning rate for LongNet (if we have it as a parameter group)
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = self.get_longnet_lr(self.base_lrs[1], epoch)

        return [group['lr'] for group in self.optimizer.param_groups]



class WarmupCosAnnlSchedlr(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_factor=0.1, warmup_factor=0.1, last_epoch=-1):
        """
        Advanced warmup cosine annealing scheduler that supports multiple parameter groups,
        each with different warmup and minimum learning rate strategies

        Args:
            optimizer (Optimizer): The optimizer to schedule
            warmup_epochs (int or list): Number of warmup epochs for each parameter group.
                                        If a single int, all groups use the same value
            total_epochs (int): Total number of training epochs
            min_lr_factor (float or list): Minimum learning rate factor for each parameter group.
                                          If a single float, all groups use the same value
            warmup_factor (float or list): Warmup starting factor for each parameter group.
                                          If a single float, all groups use the same value
            last_epoch (int): Index of the last epoch, default is -1
        """
        # Convert single values to lists to support different settings for each group
        param_groups = optimizer.param_groups
        num_groups = len(param_groups)

        if isinstance(warmup_epochs, int):
            self.warmup_epochs = [warmup_epochs] * num_groups
        else:
            assert len(warmup_epochs) == num_groups, "warmup_epochs list length must match number of parameter groups"
            self.warmup_epochs = warmup_epochs

        if isinstance(min_lr_factor, float):
            self.min_lr_factors = [min_lr_factor] * num_groups
        else:
            assert len(min_lr_factor) == num_groups, "min_lr_factor list length must match number of parameter groups"
            self.min_lr_factors = min_lr_factor

        if isinstance(warmup_factor, float):
            self.warmup_factors = [warmup_factor] * num_groups
        else:
            assert len(warmup_factor) == num_groups, "warmup_factor list length must match number of parameter groups"
            self.warmup_factors = warmup_factor

        self.total_epochs = total_epochs
        super(WarmupCosAnnlSchedlr, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            warmup_epochs = self.warmup_epochs[i]
            min_lr_factor = self.min_lr_factors[i]
            warmup_factor = self.warmup_factors[i]

            min_lr = base_lr * min_lr_factor

            if self.last_epoch < warmup_epochs:
                # Linear warmup
                # Linearly increase from base_lr * warmup_factor to base_lr
                alpha = self.last_epoch / max(1, warmup_epochs)
                warmup_lr = base_lr * warmup_factor + alpha * (base_lr - base_lr * warmup_factor)
                lrs.append(warmup_lr)
            else:
                # Cosine annealing
                progress = (self.last_epoch - warmup_epochs) / max(1, (self.total_epochs - warmup_epochs))
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                lrs.append(min_lr + (base_lr - min_lr) * cosine_factor)

        return lrs



#%%
class StepUnfreezer:
    """
    Progressively unfreeze LongNet layers during training
    Properly handles all parameter groups including non-numbered components
    """

    def __init__(self, model, initial_active_layers=None,
                 unfreeze_schedule=None, total_epochs=100,
                 max_frozen_layers=4,
                 unfreeze_head_components_with_last_layer=False):
        """
        Arguments:
        ----------
        model: nn.Module
            Model with longnetmodel
        initial_active_layers: list or None
            Initially active layers (e.g., [-1] for last layer only)
            If None, ALL layers start frozen
        unfreeze_schedule: dict
            Mapping of epoch -> layers to unfreeze
        total_epochs: int
            Total training epochs
        max_frozen_layers: int
            Maximum number of early layers to keep frozen permanently
        unfreeze_head_components_with_last_layer: bool
            If True, automatically unfreeze head components (encoder.layer_norm,
            norm, cls_token) when the last layer is unfrozen
        """
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule or {}
        self.total_epochs = total_epochs
        self.max_frozen_layers = max_frozen_layers
        self.unfreeze_head_components_with_last_layer = unfreeze_head_components_with_last_layer

        self.head_component_patterns = [
            'encoder.layer_norm.weight',
            'encoder.layer_norm.bias',
            'norm.weight',
            'norm.bias'
        ]

        # Detect number of layers
        self.num_layers = self._detect_num_layers()

        # Validate max_frozen_layers
        if max_frozen_layers >= self.num_layers:
            raise ValueError(
                f"max_frozen_layers ({max_frozen_layers}) must be less than total layers ({self.num_layers})")

        # Track head components status
        self.head_components_unfrozen = False

        # Initialize active layers
        if initial_active_layers is None:
            self.current_active_layers = []
            print("Starting with ALL LongNet layers frozen")
        else:
            self.current_active_layers = initial_active_layers.copy()
            # Check if last layer is in initial active layers
            if self._has_last_layer(initial_active_layers):
                self.head_components_unfrozen = True
            print(f"Starting with active layers: {initial_active_layers}")

        # Validate schedule
        self._validate_schedule()

        self._print_config()

        # Apply initial state
        self._apply_current_state()

    def _detect_num_layers(self):
        """Detect number of encoder layers"""
        num_layers = 0
        for name, _ in self.model.longnetmodel.named_parameters():
            if 'encoder.layers.' in name:
                layer_num = int(name.split('encoder.layers.')[1].split('.')[0])
                if layer_num >= num_layers:
                    num_layers = layer_num + 1

        if num_layers == 0:
            print("Warning: Could not detect layers, defaulting to 12")
            num_layers = 12

        return num_layers

    def _has_last_layer(self, layer_list):
        """Check if layer list contains the last layer"""
        for layer_idx in layer_list:
            positive_idx = self._convert_to_positive_idx(layer_idx)
            if positive_idx == self.num_layers - 1:
                return True
        return False

    def _print_config(self):
        """Print configuration"""
        print(f"\n{'=' * 70}")
        print(f"Progressive Unfreezing Configuration:")
        print(f"{'=' * 70}")
        print(f"  Total LongNet layers: {self.num_layers} (layers 0-{self.num_layers - 1})")
        print(f"  Permanently frozen layers: 0-{self.max_frozen_layers - 1} (first {self.max_frozen_layers} layers)")
        print(f"  Trainable layer range: {self.max_frozen_layers}-{self.num_layers - 1}")
        print(
            f"  Initial active layers: {self.current_active_layers if self.current_active_layers else 'None (all frozen)'}")
        print(f"  Unfreeze schedule: {self.unfreeze_schedule if self.unfreeze_schedule else 'None'}")
        print(f"  Auto-unfreeze head components: {self.unfreeze_head_components_with_last_layer}")
        print(f"{'=' * 70}\n")

    def _validate_schedule(self):
        """Validate that schedule doesn't try to unfreeze protected layers"""
        for epoch, layers in self.unfreeze_schedule.items():
            for layer_idx in layers:
                positive_idx = self._convert_to_positive_idx(layer_idx)
                if positive_idx < self.max_frozen_layers:
                    raise ValueError(
                        f"Unfreeze schedule at epoch {epoch} tries to unfreeze layer {layer_idx} "
                        f"(absolute: {positive_idx}), but layers 0-{self.max_frozen_layers - 1} "
                        f"are permanently frozen"
                    )

    def _convert_to_positive_idx(self, layer_idx):
        """Convert negative index to positive"""
        if layer_idx < 0:
            return self.num_layers + layer_idx
        return layer_idx

    def _freeze_all(self):
        """Freeze all LongNet parameters"""
        for param in self.model.longnetmodel.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer_idx):
        """
        Unfreeze a specific encoder layer (all its components)

        This unfreezes:
        - encoder.layers.{layer_idx}.self_attn.*
        - encoder.layers.{layer_idx}.ffn.*
        - encoder.layers.{layer_idx}.*_layer_norm.*
        """
        positive_idx = self._convert_to_positive_idx(layer_idx)

        if positive_idx < self.max_frozen_layers:
            return False  # Protected layer

        unfrozen_count = 0
        for name, param in self.model.longnetmodel.named_parameters():
            # Match: encoder.layers.{N}.{anything}
            if f'encoder.layers.{positive_idx}.' in name:
                param.requires_grad = True
                unfrozen_count += 1

        return unfrozen_count > 0

    def _unfreeze_head_components(self):
        """
        Unfreeze head components (should be trained with final layers):
        - cls_token
        - encoder.layer_norm.*
        - norm.*

        Does NOT unfreeze patch_embed (input projection - kept frozen)
        """
        if self.head_components_unfrozen:
            return

        for name, param in self.model.longnetmodel.named_parameters():
            for pattern in self.head_component_patterns:
                if pattern in name:
                    param.requires_grad = True
                    break

        self.head_components_unfrozen = True
        print("  → Head components (cls_token, encoder.layer_norm, norm) unfrozen")

    def _apply_current_state(self):
        """Apply current freezing/unfreezing state to model"""
        # Freeze everything first
        self._freeze_all()

        # Unfreeze active layers
        for layer_idx in self.current_active_layers:
            self._unfreeze_layer(layer_idx)

        # Unfreeze head components if last layer is active
        if self.unfreeze_head_components_with_last_layer:
            if self._has_last_layer(self.current_active_layers):
                self._unfreeze_head_components()

    def step(self, epoch):
        """
        Check if we need to unfreeze more layers at this epoch

        Returns:
        --------
        bool: True if layers were unfrozen (optimizer needs recreation)
        """
        if epoch not in self.unfreeze_schedule:
            return False

        new_layers = self.unfreeze_schedule[epoch]
        valid_new_layers = []

        for layer_idx in new_layers:
            positive_idx = self._convert_to_positive_idx(layer_idx)

            if positive_idx < self.max_frozen_layers:
                print(f"  ⚠ Skipping layer {layer_idx} (abs: {positive_idx}) - protected by max_frozen_layers")
                continue

            valid_new_layers.append(layer_idx)
            self.current_active_layers.append(layer_idx)

        if not valid_new_layers:
            return False

        # Apply unfreezing
        self._apply_current_state()

        # Print status
        active_positive = sorted(set([
            self._convert_to_positive_idx(l)
            for l in self.current_active_layers
        ]))

        frozen_layers = [i for i in range(self.num_layers) if i not in active_positive]

        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch}: UNFREEZING LAYERS")
        print(f"{'=' * 70}")
        print(f"  Newly unfrozen layers: {valid_new_layers}")
        print(f"  Currently active layers: {active_positive}")
        print(f"  Permanently frozen: layers {list(range(self.max_frozen_layers))}")
        print(f"  Temporarily frozen: {[i for i in frozen_layers if i >= self.max_frozen_layers]}")
        print(f"  Head components unfrozen: {self.head_components_unfrozen}")
        print(f"  Total trainable layers: {len(active_positive)}/{self.num_layers - self.max_frozen_layers}")
        print(f"{'=' * 70}\n")

        return True

    def get_trainable_params(self):
        """
        Get currently trainable LongNet parameters separated by layer

        Returns:
        --------
        list of lists: Each sublist contains parameters for one layer.
                      Head components (encoder.layer_norm, norm, cls_token) are
                      grouped with the last active layer.
                      Layers are ordered from lowest to highest index.

        Example: If layers 10, 11 are active:
            [
                [layer_10_params...],
                [layer_11_params... + head_component_params...]
            ]
        """
        # Collect parameters by layer
        layer_params = {}
        head_params = []
        other_params = []

        for name, param in self.model.longnetmodel.named_parameters():
            if not param.requires_grad:
                continue

            # Encoder layers
            if 'encoder.layers.' in name:
                layer_num = int(name.split('encoder.layers.')[1].split('.')[0])
                if layer_num not in layer_params:
                    layer_params[layer_num] = []
                layer_params[layer_num].append(param)

            # Head components (to be grouped with last layer)
            elif name in self.head_component_patterns:
                head_params.append(param)

            # Other components (patch_embed, etc.)
            else:
                other_params.append(param)

        # Build result list ordered by layer index
        result = {}
        sorted_layers = sorted(layer_params.keys())

        for i, layer_idx in enumerate(sorted_layers):
            layer_group = layer_params[layer_idx]

            # Add head components to the last layer
            if i == len(sorted_layers) - 1:
                layer_group = layer_group + head_params

            result.update({layer_idx: layer_group})

        # # Add other params as a separate group if they exist
        # if other_params:
        #     result.append(other_params)

        return result

    def get_parameter_breakdown(self):
        """
        Get detailed breakdown of parameters by component

        Returns:
        --------
        dict: Breakdown of trainable vs frozen parameters
        """
        breakdown = {
            'patch_embed': {'trainable': 0, 'total': 0},
            'cls_token': {'trainable': 0, 'total': 0},
            'encoder_layers': {},  # layer_idx -> counts
            'encoder_layer_norm': {'trainable': 0, 'total': 0},
            'norm': {'trainable': 0, 'total': 0},
        }

        # Initialize layer counts
        for i in range(self.num_layers):
            breakdown['encoder_layers'][i] = {'trainable': 0, 'total': 0}

        for name, param in self.model.longnetmodel.named_parameters():
            num_params = param.numel()
            is_trainable = param.requires_grad

            # Categorize parameter
            if 'patch_embed' in name:
                breakdown['patch_embed']['total'] += num_params
                if is_trainable:
                    breakdown['patch_embed']['trainable'] += num_params

            elif name == 'cls_token':
                breakdown['cls_token']['total'] += num_params
                if is_trainable:
                    breakdown['cls_token']['trainable'] += num_params

            elif 'encoder.layers.' in name:
                layer_num = int(name.split('encoder.layers.')[1].split('.')[0])
                breakdown['encoder_layers'][layer_num]['total'] += num_params
                if is_trainable:
                    breakdown['encoder_layers'][layer_num]['trainable'] += num_params

            elif 'encoder.layer_norm' in name:
                breakdown['encoder_layer_norm']['total'] += num_params
                if is_trainable:
                    breakdown['encoder_layer_norm']['trainable'] += num_params

            elif name.startswith('norm.'):
                breakdown['norm']['total'] += num_params
                if is_trainable:
                    breakdown['norm']['trainable'] += num_params

        return breakdown

    def print_detailed_state(self):
        """Print detailed parameter breakdown"""
        breakdown = self.get_parameter_breakdown()

        print(f"\n{'=' * 70}")
        print(f"DETAILED PARAMETER BREAKDOWN")
        print(f"{'=' * 70}")

        # Embedding components
        print(f"\n📍 Embedding Components:")
        for comp in ['patch_embed', 'cls_token']:
            t = breakdown[comp]['trainable']
            total = breakdown[comp]['total']
            status = "✓ TRAINABLE" if t > 0 else "✗ FROZEN"
            print(f"  {comp:20s}: {t:>10,}/{total:>10,} params  {status}")

        # Encoder layers
        print(f"\n📍 Encoder Layers:")
        for layer_idx in range(self.num_layers):
            t = breakdown['encoder_layers'][layer_idx]['trainable']
            total = breakdown['encoder_layers'][layer_idx]['total']

            if layer_idx < self.max_frozen_layers:
                status = "🔒 PROTECTED (permanently frozen)"
            elif t > 0:
                status = "✓ TRAINABLE"
            else:
                status = "✗ FROZEN (can be unfrozen)"

            print(f"  Layer {layer_idx:2d}: {t:>10,}/{total:>10,} params  {status}")

        # Head components
        print(f"\n📍 Head Components:")
        for comp in ['encoder_layer_norm', 'norm']:
            t = breakdown[comp]['trainable']
            total = breakdown[comp]['total']
            status = "✓ TRAINABLE" if t > 0 else "✗ FROZEN"
            print(f"  {comp:20s}: {t:>10,}/{total:>10,} params  {status}")

        # Summary
        # MODIFIED: Updated to work with new list-based params structure
        trainable_params_list = self.get_trainable_params()
        total_trainable = sum(
            p.numel()
            for param_group in trainable_params_list
            for p in (param_group if isinstance(param_group, (list, tuple)) else [param_group])
            if hasattr(p, 'numel')
        )
        total_params = sum(p.numel() for p in self.model.longnetmodel.parameters())

        print(f"\n{'=' * 70}")
        print(f"SUMMARY:")
        print(f"  Total trainable: {total_trainable:>12,} params")
        print(f"  Total LongNet:   {total_params:>12,} params")
        print(f"  Trainable ratio: {100 * total_trainable / total_params:>11.2f}%")
        print(f"{'=' * 70}\n")



def setup_layer_selective_training(model, active_layers=[-2, -1], freeze_taskhead=False):
    """
    Freeze all LongNet layers except the specified ones

    Arguments:
    ----------
    model: nn.Module
        Model with longnetmodel and taskhead
    active_layers: list of int
        Layer indices to keep trainable (negative indexing supported)
        e.g., [-1] = last layer, [-2, -1] = last 2 layers, [10, 11] = layers 10 and 11
    freeze_taskhead: bool
        Whether to freeze the task head (usually False)

    Returns:
    --------
    param_groups: list
        Parameter groups for optimizer with only active layers
    """
    if not hasattr(model, 'longnetmodel'):
        raise ValueError("Model must have 'longnetmodel' attribute")

    # Get total number of layers in LongNet
    num_layers = None
    for name, _ in model.longnetmodel.named_parameters():
        if 'encoder.layers.' in name:
            # Extract layer number from parameter name
            layer_num = int(name.split('encoder.layers.')[1].split('.')[0])
            if num_layers is None or layer_num > num_layers:
                num_layers = layer_num

    if num_layers is not None:
        num_layers += 1  # Convert to count (0-indexed to 1-indexed)
        print(f"Detected {num_layers} layers in LongNet")
    else:
        print("Warning: Could not detect number of layers, freezing may not work correctly")
        num_layers = 12  # Default assumption

    # Convert negative indices to positive
    active_layers_positive = []
    for idx in active_layers:
        if idx < 0:
            active_layers_positive.append(num_layers + idx)
        else:
            active_layers_positive.append(idx)

    print(f"Active (trainable) layers: {active_layers_positive}")
    print(f"Frozen layers: {[i for i in range(num_layers) if i not in active_layers_positive]}")

    # Freeze all LongNet parameters first
    for param in model.longnetmodel.parameters():
        param.requires_grad = False

    # Unfreeze only selected layers
    trainable_longnet_params = []
    for name, param in model.longnetmodel.named_parameters():
        # Check if this parameter belongs to an active layer
        is_active = False
        for layer_idx in active_layers_positive:
            if f'encoder.layers.{layer_idx}.' in name:
                is_active = True
                break

        # Also keep normalization and position embeddings trainable if in active layers
        # You can customize this based on your needs
        if 'norm' in name and any(f'encoder.layers.{idx}' in name for idx in active_layers_positive):
            is_active = True

        if is_active:
            param.requires_grad = True
            trainable_longnet_params.append(param)

    print(f"Trainable LongNet parameters: {sum(p.numel() for p in trainable_longnet_params):,}")

    # Handle task head
    if freeze_taskhead:
        for param in model.taskhead.parameters():
            param.requires_grad = False
        taskhead_params = []
        print("TaskHead is frozen")
    else:
        taskhead_params = list(model.taskhead.parameters())
        print(f"Trainable TaskHead parameters: {sum(p.numel() for p in taskhead_params):,}")

    # Create parameter groups
    param_groups = []
    if taskhead_params:
        param_groups.append({'params': taskhead_params, 'name': 'taskhead'})
    if trainable_longnet_params:
        param_groups.append({'params': trainable_longnet_params, 'name': 'longnet_active_layers'})

    if not param_groups:
        raise ValueError("No parameters to train! Check your layer selection.")

    return param_groups