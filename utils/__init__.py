"""
Utility functions and model wrappers
"""

__all__ = [
    'losses',
    'model_wrapper',
    'utils',
    'logger',
    'fp16_util',
    'nn',
    'cal_pinn',
    'build_unet_from_config',
    'UNetWithTimeWrapper',
]


def __getattr__(name):
    if name == 'cal_pinn':
        from .losses import cal_pinn
        return cal_pinn
    elif name == 'build_unet_from_config':
        from .model_wrapper import build_unet_from_config
        return build_unet_from_config
    elif name == 'UNetWithTimeWrapper':
        from .model_wrapper import UNetWithTimeWrapper
        return UNetWithTimeWrapper
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


