import os
import io
import argparse
import zipfile
import pandas as pd

import torch
import itertools
import numpy as np
from torch import nn

from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset
import torch.utils.tensorboard as tensorboard
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support


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
def train(model,
          train_loader,
          val_loader,
          test_loader,
          train_iters=12500,
          lr=0.01, min_lr=0.0,
          optim='sgd',
          weight_decay=0.0,
          output_dir='output',
          eval_interval=1000,
          momentum=0.0,
          model_select='best'):
    """
    Train the linear probe model.

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
    train_iters: int
        Number of training iterations
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
        Evaluation interval
    momentum: float
        Momentum
    """
    # Set the device
    device = torch.device('cuda')
    model = model.to(device)

    # set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)

    # Set the optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    print('Set the optimizer as {}'.format(optim))

    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=min_lr)

    # Set the loss function
    criterion = nn.MSELoss()

    # Set the infinite train loader
    infinite_train_loader = itertools.cycle(train_loader)

    best_mae = float('inf')
    # Train the model
    print('Start training')
    for idx, (embed, category) in enumerate(infinite_train_loader):

        if idx >= train_iters:
            break

        embed, category = embed.to(device), category.to(device)

        # Forward pass
        output = model(embed)
        loss = criterion(output, category)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (idx + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Iteration [{idx}/{train_iters}]\tLoss: {loss.item()}\tLR: {lr}')
            writer.add_scalar('Train Loss', loss.item(), idx)
            writer.add_scalar('Learning Rate', lr, idx)
        # Print the loss
        if (idx + 1) % eval_interval == 0 or (idx + 1) == train_iters:
            print(f'Start evaluating ...')
            mae, rmse, r2, _, _, _ = evaluate(model, criterion, val_loader, device)
            print(f'Val [{idx}/{train_iters}] MAE: {mae:.3f} RMSE: {rmse:.3f} R²: {r2:.3f}')
            # accuracy, f1, precision, recall, auroc, auprc = evaluate(model, criterion, val_loader, device)
            # print(f'Val [{idx}/{train_iters}] Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')

            writer.add_scalar('Val MAE', mae, idx)
            writer.add_scalar('Val RMSE', rmse, idx)
            writer.add_scalar('Val R²', r2, idx)
            writer.add_scalar('Best MAE', best_mae, idx)

            if mae < best_mae:  # Lower MAE is better
                print('Best MAE decrease from {:.3f} to {:.3f}'.format(best_mae, mae))
                best_mae = mae
                torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

    # Save the model
    torch.save(model.state_dict(), f'{output_dir}/model.pth')

    if model_select == 'best':
        val_mae = best_mae
        model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
    else:
        val_mae = mae
        model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

    # Evaluate the model
    mae, rmse, r2, _, _, _ = evaluate(model, criterion, test_loader, device)
    print(f'Test MAE: {mae:.3f} RMSE: {rmse:.3f} R²: {r2:.3f}')
    writer.add_scalar('Test MAE', mae, idx)
    writer.add_scalar('Test RMSE', rmse, idx)
    writer.add_scalar('Test R²', r2, idx)

    f = open(f'{output_dir}/results.txt', 'w')
    f.write(f'Val MAE: {val_mae:.3f}\n')
    f.write(f'Test MAE: {mae:.3f} Test RMSE: {rmse:.3f} Test R2: {r2:.3f}\n')
    f.close()


def evaluate(model, criterion, val_loader, device):
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

        for _, (embed, target) in enumerate(val_loader):
            embed, target = embed.to(device), target.to(device)

            output = model(embed)  # DON'T squeeze for multiple outputs
            loss = criterion(output, target)
            total_loss += loss.item()

            pred_gather.append(output.cpu().numpy())
            target_gather.append(target.cpu().numpy())

    pred_gather = np.concatenate(pred_gather, axis=0)  # Shape: [samples, num_variables]
    target_gather = np.concatenate(target_gather, axis=0)

    # Calculate metrics across ALL variables (averaged)
    mae = np.mean(np.abs(pred_gather - target_gather))
    rmse = np.sqrt(np.mean((pred_gather - target_gather) ** 2))

    ss_res = np.sum((target_gather - pred_gather) ** 2)
    ss_tot = np.sum((target_gather - np.mean(target_gather, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return mae, rmse, r2, 0, 0, 0

#%%
class EmbedMatcher:
    def __init__(self, sample_ids, embed_path):
        self.sample_ids = sample_ids
        self.embed_path = embed_path

    def load_embeds(self):
        if self.embed_path.endswith('.pt'):
            collated_dict = torch.load(self.embed_path)
            embed_dict = {key: collated_dict[key] for key in self.sample_ids}

        else:
            if len(glob(self.embed_path + '*.pt')) < 2:
                raise ImportError('Need either tensor dict or respective sample pt tensor files! For insufficient tensor files under the apointed directory.')

            embed_dict = {}
            for sample_id in self.sample_ids:
                path = self.embed_path + '/' + sample_id + '.pt'
                path = path.replace('//', '/')
                embed_dict[sample_id] = torch.load(path)

        return embed_dict


class EmbeddingDataset(Dataset):
    def __init__(self, dataset_csv:str, embed_path:str, split_col:str='split_col', split:str='train', id_col:str='id', type_col:list=['life_expectancy'], z_score=False):
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

        self.samples = split_df[id_col].tolist()
        self.labels = split_df[type_col].values.tolist()

        # load the embeddings
        self.matcher = EmbedMatcher(self.samples, embed_path)
        self.embeds = self.matcher.load_embeds()

        # if need to convert to z-score
        self.z_score = z_score

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample_id, duration = self.samples[index], self.labels[index]
        embed = self.embeds[sample_id]

        if self.z_score:
            # z-score normalization
            embed = (embed - embed.mean()) / embed.std()

        return embed.squeeze(0), torch.tensor(duration, dtype=torch.float32)


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