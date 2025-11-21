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

# evaluation of the model
def get_metric(labels: np.array, probs: np.array, metric, average='micro'):
    '''
    A function to calculate metrics for multilabel classification tasks.

    Arguments:
    ----------
    metric (str): the metric to calculate. Default is 'auroc'. Options are 'auroc', 'auprc', 'bacc', 'acc' and 'qwk'.
    average (str): the averaging strategy. Default is 'micro'.
    '''

    from sklearn import metrics
    '''Return the metric score based on the metric name.'''
    if metric == 'auroc':
        return metrics.roc_auc_score(labels, probs, average=average)
    elif metric == 'auprc':
        return metrics.average_precision_score(labels, probs, average=average)
    elif metric == 'bacc':
        return metrics.balanced_accuracy_score(labels, probs)
    elif metric == 'acc':
        return metrics.accuracy_score(labels, probs)
    elif metric == 'qwk':
        return metrics.cohen_kappa_score(labels, probs, weights='quadratic')
    else:
        raise ValueError('Invalid metric: {}'.format(metric))


class MakeMetrics:
    '''
    A class to calculate metrics for multilabel classification tasks.

    Arguments:
    ----------
    metric (str): the metric to calculate. Default is 'auroc'. Options are 'auroc', 'auprc', 'bacc', 'acc' and 'qwk'.
    average (str): the averaging strategy. Default is 'micro'.
    label_dict (dict): the label dictionary, mapping from label to index. Default is None.
    '''

    def __init__(self, metric='auroc', average='micro', label_dict: dict = None):
        self.metric = metric
        self.average = average
        self.label_dict = label_dict

    def get_metric(self, labels: np.array, probs: np.array):
        '''Return the metric score based on the metric name.'''
        if self.metric == 'auroc':
            return metrics.roc_auc_score(labels, probs, average=self.average)
        elif self.metric == 'auprc':
            return metrics.average_precision_score(labels, probs, average=self.average)
        elif self.metric == 'bacc':
            return metrics.balanced_accuracy_score(labels, probs)
        elif self.metric == 'acc':
            return metrics.accuracy_score(labels, probs)
        elif self.metric == 'qwk':
            return metrics.cohen_kappa_score(labels, probs, weights='quadratic')
        else:
            raise ValueError('Invalid metric: {}'.format(self.metric))

    def process_preds(self, labels: np.array, probs: np.array):
        '''Process the predictions and labels.'''
        if self.metric in ['bacc', 'acc', 'qwk']:
            return np.argmax(labels, axis=1), np.argmax(probs, axis=1)
        else:
            return labels, probs

    @property
    def get_metric_name(self):
        '''Return the metric name.'''
        if self.metric in ['auroc', 'auprc']:
            if self.average is not None:
                return '{}_{}'.format(self.average, self.metric)
            else:
                label_keys = sorted(self.label_dict.keys(), key=lambda x: self.label_dict[x])
                return ['{}_{}'.format(key, self.metric) for key in label_keys]
        else:
            return self.metric

    def __call__(self, labels: np.array, probs: np.array) -> dict:
        '''Calculate the metric based on the given labels and probabilities.
        Args:
            labels (np.array): the ground truth labels.
            probs (np.array): the predicted probabilities.'''
        # process the predictions
        labels, probs = self.process_preds(labels, probs)
        if self.metric in ['auroc', 'auprc']:
            if self.average is not None:
                return {self.get_metric_name: self.get_metric(labels, probs)}
            else:
                score = self.get_metric(labels, probs)
                return {k: v for k, v in zip(self.get_metric_name, score)}
        else:
            return {self.get_metric_name: self.get_metric(labels, probs)}

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
          model_select='best',
          outcome_type=None,
          gc_step=1):
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
    model_select: 'best' or not
        choose either the best model or the last
    outcome_type: str: cat, bin, gene, continu (continuous)
        data type of the outcome
    gc_step: int
        gradient accumulation steps
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
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    print('Set the optimizer as {}'.format(optim))

    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=min_lr)

    # Set the loss function
    if outcome_type == 'cat':
        criterion = nn.CrossEntropyLoss()
        evaluate = evaluate_cat
        best_f1 = 0
    elif outcome_type == 'gene':
        criterion = nn.BCEWithLogitsLoss()
        evaluate = evaluate_gene
        best_f1 = 0
    elif outcome_type == 'bin':
        criterion = nn.CrossEntropyLoss()
        evaluate = evaluate_cat
        best_f1 = 0
    elif outcome_type == 'continu':
        criterion = nn.MSELoss()
        evaluate = evaluate_continu
        best_mae = float('inf')
    else:
        raise ValueError('Must provide type of outcome (any one of: cat, gene, continu)!')

    # Set the infinite train loader
    infinite_train_loader = itertools.cycle(train_loader)
    # 计算有效的训练迭代次数
    effective_train_iters = train_iters * gc_step
    # Train the model
    print(f'Start training with gradient accumulation steps: {gc_step}')
    optimizer.zero_grad()  # 在循环外先清零一次


    # Train the model
    print('Start training')
    for idx, (embed, category) in enumerate(infinite_train_loader):

        if idx >= effective_train_iters:
            break

        embed, category = embed.to(device), category.to(device)

        # Forward pass
        output = model(embed)
        loss = criterion(output, category)
        # 将损失除以累积步数（保持梯度scale一致）
        loss = loss / gc_step
        loss.backward()

        # Backward pass
        if (idx + 1) % gc_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            # 计算实际的训练步数
            actual_iter = (idx + 1) // gc_step

            if (actual_iter + 1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'Iteration [{idx}/{train_iters}]\tLoss: {loss.item()}\tLR: {lr}')
                writer.add_scalar('Train Loss', loss.item(), idx)
                writer.add_scalar('Learning Rate', lr, idx)

            # Print the loss
            if (idx + 1) % eval_interval == 0 or (idx + 1) == train_iters:
                print(f'Start evaluating ...')
                if outcome_type == 'continu':
                    mae, rmse, r2, _, _, _, pred_gather, target_gather = evaluate(model, criterion, val_loader, device)
                    print(f'Val [{idx}/{train_iters}] MAE: {mae:.3f} RMSE: {rmse:.3f} R²: {r2:.3f}')
                    # accuracy, f1, precision, recall, auroc, auprc, pred_gather, target_gather = evaluate(model, criterion, val_loader, device)
                    # print(f'Val [{idx}/{train_iters}] Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')

                    writer.add_scalar('Val MAE', mae, idx)
                    writer.add_scalar('Val RMSE', rmse, idx)
                    writer.add_scalar('Val R²', r2, idx)
                    writer.add_scalar('Best MAE', best_mae, idx)

                    if mae < best_mae:  # Lower MAE is better
                        print('Best MAE decrease from {:.3f} to {:.3f}'.format(best_mae, mae))
                        best_mae = mae
                        torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

                else:
                    accuracy, f1, precision, recall, auroc, auprc, pred_gather, target_gather = evaluate(model, criterion, val_loader, device)
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

    if outcome_type == 'continu':
        if model_select == 'best':
            val_mae = best_mae
            model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
        else:
            val_mae = mae
            model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

        # Evaluate the model
        mae, rmse, r2, _, _, _, pred_gather, target_gather = evaluate(model, criterion, test_loader, device)
        print(f'Test MAE: {mae:.3f} RMSE: {rmse:.3f} R²: {r2:.3f}')
        writer.add_scalar('Test MAE', mae, idx)
        writer.add_scalar('Test RMSE', rmse, idx)
        writer.add_scalar('Test R²', r2, idx)

        f = open(f'{output_dir}/results.txt', 'w')
        f.write(f'Val MAE: {val_mae:.3f}\n')
        f.write(f'Test MAE: {mae:.3f} Test RMSE: {rmse:.3f} Test R2: {r2:.3f}\n')
        f.close()

    else:
        if model_select == 'best':
            val_f1 = best_f1
            model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
        else:
            val_f1 = f1
            model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

        # Evaluate the model
        accuracy, f1, precision, recall, auroc, auprc, pred_gather, target_gather = evaluate(model, criterion, test_loader, device)
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

    return pred_gather, target_gather



#%%
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
    pred_binary = (pred_gather > 0.5).astype(int)  # Threshold predictions
    accuracy = (pred_binary == category_gather).mean()  # Simple element-wise accuracy
    # calculate the weighted f1 score
    f1 = f1_score(category_gather, pred_binary, average='macro', zero_division=0)
    # calculate the precision and recall, precision is not accuracy
    precision, recall, _, _ = precision_recall_fscore_support(category_gather, pred_binary, average='macro', zero_division=0)

    auroc = roc_auc_score(category_gather, torch.sigmoid(torch.tensor(pred_gather)).numpy(), average='macro')
    auprc = average_precision_score(category_gather, torch.sigmoid(torch.tensor(pred_gather)).numpy(), average='macro')

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

    return mae, rmse, r2, 0, 0, 0, pred_gather, target_gather

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
        embed = self.embeds[sample_id]
        embed = [embed[i] for i in self.feat_layer]
        embed = torch.cat(embed, dim=-1)

        if self.z_score:
            # z-score normalization
            embed = (embed - embed.mean()) / embed.std()

        # convert the label to index
        if self.outcome_type == 'cat':
            category = self.label_dict[category]
        else:
            category = torch.tensor(category, dtype=torch.float32)
        embed = embed.squeeze(0)

        return embed, category


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