from functools import partial

import torch
import torch.nn as nn
import numpy as np
import os

import timm
from timm.models.registry import register_model

from .torchscale.model import LongNetConfig as longnet_arch
from .torchscale.architecture.config import EncoderConfig
from .torchscale.architecture.decoder import Decoder, DecoderLayer
from .torchscale.architecture.encoder import Encoder, EncoderLayer
from .torchscale.component.dilated_attention import DilatedAttention
from fairscale.nn import checkpoint_wrapper, wrap



# %%
class LongNetDecoderLayer(DecoderLayer):

    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

class LongNetDecoder(Decoder):

    def build_decoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = LongNetDecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer


class LongNetEncoderLayer(EncoderLayer):

    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )


class LongNetEncoder(Encoder):

    def build_encoder_layer(
            self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = LongNetEncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer


def make_longnet_from_name(config_name: str,
                           dilated_ratio: str = '[1, 2, 4, 8, 16]',
                           segment_length: str = '[1024, 2048, 4096, 8192, 16384]',
                           drop_path_rate: int = 0.1,
                           dropout: float = 0.1):
    '''
    make LongNet model from config name

    Arguments:
    ----------
    config_name: str
        name of the config
    dilated_ratio: str
        dilated ratio
    segment_length: str
        segment length
    drop_path_rate: int
        drop path rate
    dropout: float
        dropout rate
    '''
    if config_name in longnet_arch.__dict__.keys():
        longnet_args = longnet_arch.__dict__[config_name]

    longnet_args['dropout'] = dropout
    longnet_args['drop_path_rate'] = drop_path_rate

    # set dilated ratio and segment length
    longnet_args['dilated_ratio'] = dilated_ratio
    longnet_args['segment_length'] = segment_length

    print('dilated_ratio: ', dilated_ratio)
    print('segment_length: ', segment_length)

    longnet_args = EncoderConfig(**longnet_args)
    model = LongNetEncoder(longnet_args)
    print('Number of trainable LongNet parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# %%
class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
            self,
            in_chans=1536,
            embed_dim=768,
            norm_layer=None,
            bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# %%
class LongNetViT(nn.Module):
    """
    Backbone of Vision Transformer for downstream tasks

    Arguments:
    ----------
    in_chans: int
        The number of input channels, should be the tile encoding dimension 1536.
    embed_dim: int
        The embedding dimension of the LongNet model.
    depth: int
        The number of LongNet layers in the LongNet model.
    slide_ngrids: int
        The number of grids in the slide.
    tile_size: int
        The tile size. Default is 256px.
    max_wsi_size: int
        The maximum size of the WSI.
    norm_layer: nn.LayerNorm
        The normalization layer used in the model.
    global_pool: bool
        Whether to use global pooling or not.
    dropout: float
        The dropout rate used in the model.
    drop_path_rate: float
        The drop path rate used in the model.
    """

    def __init__(self,
                 in_chans=1536,
                 embed_dim=256,
                 depth=12,
                 slide_ngrids=1000,
                 tile_size=256,
                 max_wsi_size=262144,
                 norm_layer=nn.LayerNorm,
                 global_pool=False,
                 dropout=0.25,
                 drop_path_rate=0.1,
                 **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(in_chans, embed_dim)

        self.tile_size = tile_size
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_embed', torch.zeros(1, num_patches + 1, embed_dim),
                             persistent=False)  # fixed sin-cos embedding

        self.encoder_name = "LongNet_{}_layers_{}_dim".format(depth, embed_dim)
        if kwargs.get("mlp_ratio", 4.0) != 4.0:
            self.encoder_name += "_mlp{}".format(kwargs.get("mlp_ratio"))

        # get optimal segment length
        segment_length = self.get_optimal_segment_length(max_wsi_size, tile_size)
        self.encoder = make_longnet_from_name(self.encoder_name, drop_path_rate=drop_path_rate, dropout=dropout,
                                              segment_length=segment_length)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.global_pool = global_pool
        print("Global Pooling:", self.global_pool)

        self.initialize_vit_weights()

    def initialize_vit_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def get_optimal_segment_length(self, max_wsi_size: int = 262144, tile_size: int = 256) -> str:
        '''
        Get the optimal segment length based on the maximum image size and tile size.

        Arguments:
        ----------
        max_wsi_size: int
            The maximum size of the WSI.
        tile_size: int
            The tile size.
        '''
        max_seq_len = (max_wsi_size // tile_size) ** 2
        # calculate the segment length
        segment_length = np.linspace(np.log2(1024), int(np.log2(max_seq_len)), 5)
        segment_length = np.power(2, segment_length).astype(int)
        # convert to str format
        segment_length = str(list(segment_length))
        return segment_length

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def coords_to_pos(self, coords, tile_size: int = 256):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [N, L]
        """
        coords_ = torch.floor(coords / tile_size)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + 1  # add 1 for the cls token

    def forward(self, x, coords, all_layer_embed=False):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        all_layer_embed: bool
            Whether to return embeddings from all layers or not
        """
        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords, self.tile_size)  # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if all_layer_embed:
            x_list = self.encoder(src_tokens=None, token_embeddings=x, return_all_hiddens=all_layer_embed)[
                "encoder_states"]
        else:
            x_list = [self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]]

        outcomes = []
        for x in x_list:
            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1)  # global average pooling
                outcome = self.norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]
            outcomes.append(outcome)

        return outcomes


#%%
def glio_create_model(local_path: str, model_arch: str, in_chans: int, local_dir: str = os.path.join(os.path.expanduser("~"), ".cache/"), **kwargs):
    model = timm.create_model(model_arch, pretrained=False, in_chans=in_chans, **kwargs)

    if os.path.exists(local_path):
        state_dict = torch.load(local_path, map_location="cpu")["model"]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            for k in missing_keys:
                print("Missing ", k)

        if len(unexpected_keys) > 0:
            for k in unexpected_keys:
                print("Unexpected ", k)

        print("\033[92m Successfully Loaded Pretrained GigaPath model from {} \033[00m".format(local_path))
    else:
        print("\033[93m Pretrained weights not found at {}. Randomly initialized the model! \033[00m".format(local_path))

    return model

# %%
@register_model
def gigapath_slide_enc12l768d(**kwargs):
    model = LongNetViT(embed_dim=768, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def gigapath_slide_enc24l1024d(**kwargs):
    model = LongNetViT(embed_dim=1024, depth=24, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def gigapath_slide_enc12l1536d(**kwargs):
    model = LongNetViT(embed_dim=1536, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def gigapath_slide_enc5l64d(**kwargs):
    model = LongNetViT(embed_dim=64, depth=5, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def gigapath_slide_enc6l128d(**kwargs):
    model = LongNetViT(embed_dim=128, depth=6, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def gigapath_slide_enc3l32d(**kwargs):
    model = LongNetViT(embed_dim=32, depth=3, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model