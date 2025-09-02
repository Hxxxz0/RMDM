"""
PhyRMDM: Physics-informed Radio Map Diffusion Model

A physics-informed radio map diffusion model project
"""

__version__ = "1.0.0"
__author__ = "PhyRMDM Team"

__all__ = [
    'utils',
    'lib',
    'build_unet_from_config',
    'cal_pinn',
]

# Lazy import to avoid dependency issues
def __getattr__(name):
    if name == 'utils':
        from . import utils
        return utils
    elif name == 'lib':
        from . import lib
        return lib
    elif name == 'build_unet_from_config':
        from .utils import build_unet_from_config
        return build_unet_from_config
    elif name == 'cal_pinn':
        from .utils import cal_pinn
        return cal_pinn
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
