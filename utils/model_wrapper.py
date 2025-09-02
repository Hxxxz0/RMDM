import torch
import torch.nn as nn
import os
import sys

# Set up path to import UNet model
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from unet import UNetModel_newpreview


class UNetWithTimeWrapper(nn.Module):
    """
    Wrap existing UNetModel_newpreview as standard diffusion model:
    - Input: noisy_input (B, C, H, W) containing conditions and noisy target; timesteps (B,)
    - Output: noise_pred (B, out_ch, H, W) predicted noise
    """
    def __init__(self, unet: UNetModel_newpreview):
        super().__init__()
        self.unet = unet

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor):
        # Return underlying UNet output directly; preserve both branches (noise and cal) if tuple
        out = self.unet(sample, timesteps)
        return out


def build_unet_from_config(cfg: dict) -> UNetWithTimeWrapper:
    # Handle empty channel_mult, set default based on image_size
    if not cfg['channel_mult']:
        if cfg['image_size'] == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif cfg['image_size'] == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif cfg['image_size'] == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif cfg['image_size'] == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {cfg['image_size']}")
    else:
        channel_mult = tuple(int(x) for x in cfg['channel_mult'].split(','))

    unet = UNetModel_newpreview(
        image_size=cfg['image_size'],
        in_channels=cfg['in_ch'],
        model_channels=cfg['num_channels'],
        out_channels=cfg.get('out_ch', 1),  # Use configured output channels
        num_res_blocks=cfg['num_res_blocks'],
        attention_resolutions=tuple(int(cfg['image_size']) // int(r) for r in cfg['attention_resolutions'].split(',')),
        dropout=cfg['dropout'],
        channel_mult=channel_mult,
        num_classes=(2 if cfg.get('class_cond', False) else None),
        use_checkpoint=cfg['use_checkpoint'],
        use_fp16=cfg['use_fp16'],
        num_heads=cfg['num_heads'],
        num_head_channels=cfg['num_head_channels'],
        num_heads_upsample=cfg['num_heads_upsample'],
        use_scale_shift_norm=cfg['use_scale_shift_norm'],
        resblock_updown=cfg['resblock_updown'],
        use_new_attention_order=cfg['use_new_attention_order'],
        high_way=True,
    )
    return UNetWithTimeWrapper(unet)


