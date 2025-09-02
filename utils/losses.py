import torch
import torch.nn.functional as F


def cal_pinn(cal, buildings, shooter, k=1.0, k_building=1.0):
    """
    Migrated from guided_diffusion.gaussian_diffusion.cal_pinn, behavior unchanged:
    - cal, buildings, shooter: (B, H, W)
    - Returns: (B,) per-sample PINN loss
    """
    cal_t = cal.unsqueeze(1)
    buildings_t = buildings.unsqueeze(1)
    shooter_t = shooter.unsqueeze(1)

    device = cal_t.device
    dtype = cal_t.dtype

    lap_kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0,-4.0, 1.0],
                               [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1,1,3,3)
    lap = F.conv2d(cal_t, lap_kernel, padding=1)

    buildings_mask = (buildings_t > 0.5)
    shooter_mask = (shooter_t > 0.5)

    k_tensor = torch.tensor(float(k), device=device, dtype=dtype)
    k_building_tensor = torch.tensor(float(k_building), device=device, dtype=dtype)
    k_map = torch.where(buildings_mask, k_building_tensor, k_tensor)

    residual = lap + (k_map ** 2) * cal_t
    L_pde = residual.pow(2).flatten(1).mean(1)

    bc_num = buildings_mask.sum(dim=(1,2,3)).clamp_min(1)
    L_bc = (cal_t.pow(2) * buildings_mask).sum(dim=(1,2,3)) / bc_num

    src_num = shooter_mask.sum(dim=(1,2,3)).clamp_min(1)
    L_source = ((cal_t - 1.0).pow(2) * shooter_mask).sum(dim=(1,2,3)) / src_num

    loss = L_pde + L_bc + L_source
    return loss


