import os
import timm
from . import longn


#%%
def giga_tile_enc(path: str, checkpoint_file='/pytorch_model.bin', verbose=False):
    tile_encoder = timm.create_model(f"local-dir:{path}", pretrained=False,
                                     checkpoint_path=path + checkpoint_file)
    if verbose:
        print("Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))
    return tile_encoder


def giga_slide_enc(path: str='', global_pool=False):
    print(os.getcwd())
    slide_encoder_model = longn.glio_create_model(local_path=path, model_arch="gigapath_slide_enc12l768d", in_chans=1536, global_pool=global_pool)
    print("Slide encoder param #", sum(p.numel() for p in slide_encoder_model.parameters()))

    return slide_encoder_model
