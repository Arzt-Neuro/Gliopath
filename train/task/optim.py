# the following code / functions were mostly from utils.py (finetune)


# cross validation: keyword: fold
## script from the authors did not provide cross validation but only random status (random seed) represented by [seed]

# log writer:
## define: utils
## apply: training

# fp16 scaler:
## how to implant: training
## need to define elsewhere

# weighted sampler
from torch.utils.data import Dataset, WeightedRandomSampler, RandomSampler
weighted_sample = True
train_dataset = Dataset(...)
if weighted_sample and not task_config.get('setting', 'multi_class') == 'multi_label':
    # get the weights for each class, we only do this for multi-class classification
    N = len(train_dataset)
    weights = {}
    for idx in range(N):
        label = int(train_dataset.labels[idx][0])
        if label not in weights: weights[label] = 0
        weights[label] += 1.0 / N
    for l in weights.keys(): weights[l] = 1.0 / weights[l]
    sample_weights = [weights[int(train_dataset.labels[i][0])] for i in range(N)]
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
else:
    train_sampler = RandomSampler(train_dataset)


# padding position
## from a slide, there could be different number of patches. this function means to make the patch number consistent by padding
def pad_tensors(imgs, coords):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/tree/main
    # ------------------------------------------------------------------------------------------
    max_len = max([t.size(0) for t in imgs])  # get the maximum length
    padded_tensors = []  # list to store all padded tensors
    padded_coords = []  # list to store all padded coords
    masks = []  # list to store all masks
    for i in range(len(imgs)):
        # tensor: [L, d]
        tensor = imgs[i]
        # coords: [L, 2]
        coord = coords[i]
        N_i = tensor.size(0)  # get the original length
        # create a new tensor of shape (max_len, d) filled with zeros
        padded_tensor = torch.zeros(max_len, tensor.size(1))
        padded_coord = torch.zeros(max_len, 2)
        # create a new tensor of shape (max_len) filled with zeros for mask
        mask = torch.zeros(max_len)
        # place the original tensor into the padded tensor
        padded_tensor[:N_i] = tensor
        padded_coord[:N_i] = coord
        # the mask is filled with ones at the same indices as the original tensor
        mask[:N_i] = torch.ones(N_i)
        padded_tensors.append(padded_tensor)
        padded_coords.append(padded_coord)
        masks.append(mask)

    # concatenate all tensors along the 0th dimension
    padded_tensors = torch.stack(padded_tensors)
    padded_coords = torch.stack(padded_coords)
    masks = torch.stack(masks)
    # convert masks to bool type
    masks = masks.bool()
    return padded_tensors, padded_coords, masks


def slide_collate_fn(samples):
    '''Separate the inputs and targets into separate lists
    Return value {imgs: [N, L, 256, 384], pad_mask: [N, L]}'''
    image_list = [s['imgs'] for s in samples]
    img_len_list = [s['imgs'].size(0) for s in samples]
    coord_list = [s['coords'] for s in samples]
    label_list = [s['labels'] for s in samples]
    slide_id_list = [s['slide_id'] for s in samples]
    labels = torch.stack(label_list)
    pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)

    data_dict = {'imgs': pad_imgs,
                 'img_lens': img_len_list,
                 'coords': pad_coords,
                 'slide_id': slide_id_list,
                 'pad_mask': pad_mask,
                 'labels': labels}
    return data_dict

#%%
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    # ------------------------------------------------------------------------------------------
    param_group_names = {}
    param_groups = {}

    num_layers = model.slide_encoder.encoder.num_layers + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if 'mask_token' in n or 'slide_encoder.decoder' in n:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id(n, num_layers)

        group_name = n + "_%d_%s" % (layer_id + 1, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id(name, num_layers):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    # ------------------------------------------------------------------------------------------
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('slide_encoder.encoder.layers'):
        return int(name.split('.')[3]) + 1
    else:
        return num_layers


def adjust_learning_rate(optimizer, epoch, args):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
    # ------------------------------------------------------------------------------------------
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_optimizer(args, model):
    '''Set up the optimizer for the model.'''
    param_groups = param_groups_lrd(model, args.optim_wd,
                                    layer_decay=args.layer_decay)
    # make the optimizer
    optim_func = torch.optim.AdamW if args.optim == 'adamw' else torch.optim.Adam
    optimizer = optim_func(param_groups, lr=args.lr)

    return optimizer


def get_loss_function(task_config: dict):
    '''Get the loss function based on the task configuration.'''
    task_setting = task_config.get('setting', 'multi_class')
    if task_setting == 'multi_label':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif task_setting == 'multi_class' or task_setting == 'binary':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return loss_fn


def get_records_array(record_len: int, n_classes) -> dict:
    '''Get the records array based on the task configuration.'''
    record = {
        'prob': np.zeros((record_len, n_classes), dtype=np.float32),
        'label': np.zeros((record_len, n_classes), dtype=np.float32),
        'loss': 0.0,
    }
    return record

class Monitor_Score:
    # ------------------------------------------------------------------------------------------
    # References:
    # MCAT: https://github.com/mahmoodlab/MCAT/blob/master/utils/core_utils.py
    # ------------------------------------------------------------------------------------------
    def __init__(self):
        self.best_score = None

    def __call__(self, val_score, model, ckpt_name:str='checkpoint.pt'):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def log_writer(log_dict: dict, step: int, report_to: str='tensorboard', writer=None):
    '''Log the dictionary to the writer.'''
    if report_to == 'tensorboard':
        for k, v in log_dict.items():
            writer.add_scalar(k, v, step)
    elif report_to == 'wandb':
        writer.log(log_dict, step=step)
    else:
        raise NotImplementedError