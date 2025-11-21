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
          train_iters=4000,
          lr=0.01, min_lr=0.0,
          optim='sgd',
          weight_decay=0.0,
          output_dir='output',
          eval_interval=100,
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
    criterion = nn.CrossEntropyLoss()

    # Set the infinite train loader
    infinite_train_loader = itertools.cycle(train_loader)

    best_f1 = 0
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
            accuracy, f1, precision, recall, auroc, auprc = evaluate(model, criterion, val_loader, device)
            print(f'Val [{idx}/{train_iters}] Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')

            writer.add_scalar('Val Accuracy', accuracy, idx)
            writer.add_scalar('Val f1', f1, idx)
            writer.add_scalar('Val AUROC', auroc, idx)
            writer.add_scalar('Val AUPRC', auprc, idx)
            writer.add_scalar('Val Precision', precision, idx)
            writer.add_scalar('Val Recall', recall, idx)
            writer.add_scalar('Best f1', best_f1, idx)

            if f1 > best_f1:
                print('Best f1 increase from {} to {}'.format(best_f1, f1))
                best_f1 = f1
                torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

    # Save the model
    torch.save(model.state_dict(), f'{output_dir}/model.pth')

    if model_select == 'best':
        val_f1 = best_f1
        model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
    else:
        val_f1 = f1
        model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

    # Evaluate the model
    accuracy, f1, precision, recall, auroc, auprc = evaluate(model, criterion, test_loader, device)
    print(f'Test Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')
    writer.add_scalar('Test Accuracy', accuracy, idx)
    writer.add_scalar('Test f1', f1, idx)
    writer.add_scalar('Test AUROC', auroc, idx)
    writer.add_scalar('Test AUPRC', auprc, idx)
    writer.add_scalar('Test Precision', precision, idx)
    writer.add_scalar('Test Recall', recall, idx)

    f = open(f'{output_dir}/results.txt', 'w')
    f.write(f'Val f1: {val_f1}\n')
    f.write(f'Test f1: {f1} Test AUROC: {auroc} Test AUPRC: {auprc}\n')
    f.close()


def evaluate(model, criterion, val_loader, device):
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
        for _, (embed, category) in enumerate(val_loader):

            embed, category = embed.to(device), category.to(device)

            # forward pass
            output = model(embed)
            loss = criterion(output, category)
            total_loss += loss.item()
            # gather the predictions and categories
            pred_gather.append(output.cpu().numpy())
            category_gather.append(category.cpu().numpy())

    # calculate the accuracy, AUROC, AUPRC
    pred_gather = np.concatenate(pred_gather)
    category_gather = np.concatenate(category_gather)
    accuracy = (pred_gather.argmax(1) == category_gather).mean()
    # calculate the weighted f1 score
    f1 = f1_score(category_gather, pred_gather.argmax(1), average='weighted')
    # calculate the precision and recall, precision is not accuracy
    precision, recall, _, _ = precision_recall_fscore_support(category_gather, pred_gather.argmax(1), average='macro')

    auroc = roc_auc_score(to_onehot(category_gather, pred_gather.shape[1]), pred_gather, average='macro')
    auprc = average_precision_score(to_onehot(category_gather, pred_gather.shape[1]), pred_gather, average='macro')

    return accuracy, f1, precision, recall, auroc, auprc

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
    def __init__(self, dataset_csv:str, embed_path:str, split_col:str='split_col', split:str='train', id_col:str='id', type_col:str='tumour_type', z_score=False):
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
        self.labels = split_df[type_col].tolist()

        # load the embeddings
        self.matcher = EmbedMatcher(self.samples, embed_path)
        self.embeds = self.matcher.load_embeds()

        # generate a dict for labels
        label_set = list(self.labels)
        label_set = sorted(set(label_set))
        self.label_dict = {label: i for i, label in enumerate(label_set)}

        # if need to convert to z-score
        self.z_score = z_score

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample_id, category = self.samples[index], self.labels[index]
        embed = self.embeds[sample_id]

        if self.z_score:
            # z-score normalization
            embed = (embed - embed.mean()) / embed.std()

        # convert the label to index
        category = self.label_dict[category]

        return embed.squeeze(0), category


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