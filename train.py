import os
import re
import argparse
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from diffusers import DDPMScheduler

# Internal module imports
from utils import build_unet_from_config, cal_pinn
from lib import loaders as radio_loaders


def build_dataloader(data_name, data_dir, image_size, batch_size, workers):
    if data_name == 'Radio':
        ds = radio_loaders.RadioUNet_c(phase="train", dir_dataset=data_dir)
        in_ch = 2  # [buildings, Tx] input channels
        out_ch = 1  # target output channels
    elif data_name == 'Radio_2':
        ds = radio_loaders.RadioUNet_s(phase="train", carsSimul="yes", carsInput="yes", dir_dataset=data_dir)
        in_ch = 4  # [buildings, Tx, samples, cars] input channels
        out_ch = 1  # target output channels
    elif data_name == 'Radio_3':
        ds = radio_loaders.RadioUNet_s(phase="train", simulation="rand", cityMap="missing", missing=4, dir_dataset=data_dir)
        in_ch = 3  # [buildings, Tx, samples] input channels
        out_ch = 1  # target output channels
    else:
        raise ValueError("data_name must be 'Radio' | 'Radio_2' | 'Radio_3'")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    return dl, in_ch, out_ch


def parse_args():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--data_name', type=str, default='Radio', choices=['Radio','Radio_2','Radio_3'])
    parser.add_argument('--data_dir', type=str, required=True, help='RadioMapSeer root directory, e.g., /path/to/RadioMapSeer/')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    # Model arguments
    parser.add_argument('--num_channels', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attention_resolutions', type=str, default='16,8')
    parser.add_argument('--channel_mult', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_checkpoint', type=bool, default=False)
    parser.add_argument('--use_scale_shift_norm', type=bool, default=True)
    parser.add_argument('--resblock_updown', type=bool, default=False)
    parser.add_argument('--use_fp16', type=bool, default=False)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_head_channels', type=int, default=-1)
    parser.add_argument('--num_heads_upsample', type=int, default=-1)
    parser.add_argument('--use_new_attention_order', type=bool, default=False)
    # Diffusion/Training arguments
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='linear', choices=['linear','cosine'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no','fp16','bf16'])
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_phy')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save every N steps, 0 means save only at the end')
    # Resume training
    parser.add_argument('--resume_from', type=str, default='', help='Model checkpoint path, e.g., /path/to/model_phy_step10000.pth')
    parser.add_argument('--resume_step', type=int, default=0, help='Step to resume from (0 to infer from filename)')
    return parser.parse_args()


def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    dl, in_ch, out_ch = build_dataloader(args.data_name, args.data_dir, args.image_size, args.batch_size, args.workers)

    # Build UNet configuration - input: conditions + noisy target, output: denoised result
    cfg = {
        'image_size': args.image_size,
        'in_ch': in_ch + out_ch,  
        'out_ch': out_ch,  
        'num_channels': args.num_channels,
        'num_res_blocks': args.num_res_blocks,
        'channel_mult': args.channel_mult,
        'num_heads': args.num_heads,
        'num_head_channels': args.num_head_channels,
        'num_heads_upsample': args.num_heads_upsample,
        'attention_resolutions': args.attention_resolutions,
        'dropout': args.dropout,
        'class_cond': False,
        'use_checkpoint': args.use_checkpoint,
        'use_scale_shift_norm': args.use_scale_shift_norm,
        'resblock_updown': args.resblock_updown,
        'use_fp16': args.use_fp16,
        'use_new_attention_order': args.use_new_attention_order,
        'learn_sigma': False,
    }
    model = build_unet_from_config(cfg)

    # Load checkpoint if resume path is specified (before prepare for single/multi-GPU compatibility)
    start_step = 0
    if getattr(args, 'resume_from', ''):
        ckpt_path = args.resume_from
        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location='cpu')
            try:
                model.load_state_dict(state, strict=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}\n{e}")
            # Infer starting step: use --resume_step if specified, otherwise extract from filename
            if getattr(args, 'resume_step', 0) and args.resume_step > 0:
                start_step = args.resume_step
            else:
                m = re.search(r"model_phy_step(\d+)\.pth", os.path.basename(ckpt_path))
                if m:
                    start_step = int(m.group(1))
            print(f"[Resume] Loaded checkpoint from {ckpt_path}, starting step={start_step}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Use diffusers scheduler for noise scheduling
    beta_schedule = 'linear' if args.noise_schedule == 'linear' else 'squaredcos_cap_v2'
    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps,
                              beta_schedule=beta_schedule,
                              prediction_type='epsilon')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    
    # Set find_unused_parameters=True for distributed training
    if hasattr(model, 'module'):
        # This is a DDP wrapped model
        model.find_unused_parameters = True
    
    model.train()

    step = start_step
    while step < args.max_steps:
        for inputs, image_gain, _ in dl:
            # Separate condition inputs and target
            if image_gain.dim() == 3:
                image_gain = image_gain.unsqueeze(1)
            
            # Condition inputs (buildings, antennas, etc.)
            # Cache original masks for PINN (unmodified)
            raw_buildings = inputs[:, 0, ...].to(device)
            if inputs.size(1) >= 2:
                raw_antenna = inputs[:, 1, ...].to(device)
            else:
                raw_antenna = torch.zeros_like(raw_buildings)

            conditions = inputs.to(device)  # (B, in_ch, H, W)
            # Consistent with RMDM: linear combination on channel 0: ch0 += 10 * ch1 (when ch1 exists)
            if conditions.size(1) >= 2:
                conditions[:, 0, ...] = conditions[:, 0, ...] + 10.0 * conditions[:, 1, ...]
            # Target (signal strength map)
            target_clean = image_gain.to(device)  # (B, 1, H, W)
            
            b = target_clean.shape[0]
            t = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device).long()

            # Standard diffusion: add noise to target
            noise = torch.randn_like(target_clean)
            target_noisy = scheduler.add_noise(target_clean, noise, t)
            
            # Concatenate conditions and noisy target as model input
            model_input = torch.cat([conditions, target_noisy], dim=1)  # (B, in_ch+1, H, W)

            # Forward pass: enforce RMDM consistency, model must return (pred_noise, cal)
            out = model(model_input, t)
            if not isinstance(out, tuple) or len(out) < 2:
                raise RuntimeError("Model must output (pred_noise, cal) for RMDM consistency. Please enable cal branch output.")
            pred_noise, cal = out[0], out[1]

            # Diffusion loss: predicted noise vs ground truth noise
            loss_diff = F.mse_loss(pred_noise, noise)

            # Cal reconstruction supervision (RMDM consistent)
            loss_cal_recon = F.mse_loss(cal, target_clean)

            # PINN loss (applied to cal, RMDM consistent)
            # Use unmodified original masks to avoid linear combination affecting PINN masks
            buildings = raw_buildings
            antenna = raw_antenna
            loss_pinn_vec = cal_pinn(cal[:, 0, :, :], buildings, antenna, k=0.2)
            loss_pinn = loss_pinn_vec.mean()

            # Total loss (RMDM structure aligned): diffusion + reconstruction + PINN
            loss = loss_diff + loss_cal_recon + loss_pinn

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_main_process and step % args.log_interval == 0:
                print(f"step {step} loss {loss.item():.4f} diff {loss_diff.item():.4f} cal {loss_cal_recon.item():.4f} pinn {loss_pinn.item():.4f}")

            # Periodic saving: main process only, avoid saving at step=0
            if accelerator.is_main_process and args.save_interval and args.save_interval > 0:
                if step > 0 and (step % args.save_interval == 0):
                    os.makedirs(args.save_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    ckpt_path = os.path.join(args.save_dir, f'model_phy_step{step}.pth')
                    torch.save(unwrapped_model.state_dict(), ckpt_path)

            step += 1
            if step >= args.max_steps:
                break

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(args.save_dir, 'model_phy.pth'))


if __name__ == '__main__':
    main()


