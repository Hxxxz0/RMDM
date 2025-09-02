#!/usr/bin/env python3
"""
Full test set inference and evaluation script - supports DDPM and DDIM
Compute metrics: MSE, NMSE, SSIM, PSNR
"""

import os
import argparse
import csv
import time
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, DDIMScheduler

# Internal module imports
from utils import build_unet_from_config
from lib import loaders as radio_loaders


# ============================================================================
# Data loading functions
# ============================================================================

def build_dataloader(data_name: str, data_dir: str, image_size: int, 
                    batch_size: int, workers: int, phase: str = "test") -> Tuple[DataLoader, int, int]:
    """Build data loader"""
    dataset_configs = {
        'Radio': {
            'loader': lambda: radio_loaders.RadioUNet_c(phase=phase, dir_dataset=data_dir),
            'in_ch': 2,   # [buildings, Tx]
            'out_ch': 1   # target
        },
        'Radio_2': {
            'loader': lambda: radio_loaders.RadioUNet_s(
                phase=phase, carsSimul="yes", carsInput="yes", dir_dataset=data_dir
            ),
            'in_ch': 4,   # [buildings, Tx, samples, cars]
            'out_ch': 1   # target
        },
        'Radio_3': {
            'loader': lambda: radio_loaders.RadioUNet_s(
                phase=phase, simulation="rand", cityMap="missing", missing=4, dir_dataset=data_dir
            ),
            'in_ch': 3,   # [buildings, Tx, samples]
            'out_ch': 1   # target
        }
    }
    
    if data_name not in dataset_configs:
        raise ValueError(f"data_name must be one of {list(dataset_configs.keys())}")
    
    config = dataset_configs[data_name]
    ds = config['loader']()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, 
                   num_workers=workers, pin_memory=True)
    
    return dl, config['in_ch'], config['out_ch']


# ============================================================================
# Scheduler functions
# ============================================================================

def create_scheduler(scheduler_type: str, num_train_timesteps: int = 1000, 
                    noise_schedule: str = 'linear'):
    """Create scheduler"""
    beta_schedule = 'linear' if noise_schedule == 'linear' else 'squaredcos_cap_v2'
    
    if scheduler_type == 'ddpm':
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type='epsilon'
        )
    elif scheduler_type == 'ddim':
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type='epsilon',
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=0
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


# ============================================================================
# Inference functions
# ============================================================================

def sample_ddpm(model, scheduler, conditions: torch.Tensor, 
               num_inference_steps: int = 1000, device: str = 'cuda') -> Tuple[torch.Tensor, float]:
    """DDPM sampling"""
    batch_size, _, height, width = conditions.shape
    
    # Initial noise
    image = torch.randn((batch_size, 1, height, width), device=device)
    
    # Set inference steps
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            # Concatenate conditions and current noisy image
            model_input = torch.cat([conditions, image], dim=1)
            
            # Predict noise
            timestep_batch = torch.tensor([timestep] * batch_size, device=device).long()
            out = model(model_input, timestep_batch)
            noise_pred = out[0] if isinstance(out, tuple) else out
            
            # DDPM denoising step
            batch_size_curr = image.shape[0]
            image_list = []
            for j in range(batch_size_curr):
                single_step = scheduler.step(
                    noise_pred[j:j+1], timestep, image[j:j+1], return_dict=False
                )[0]
                image_list.append(single_step)
            image = torch.cat(image_list, dim=0)
    
    end_time = time.time()
    sampling_time = end_time - start_time
    
    return image, sampling_time


def sample_ddim(model, scheduler, conditions: torch.Tensor, 
               num_inference_steps: int = 50, device: str = 'cuda', 
               ddim_eta: float = 1.0) -> Tuple[torch.Tensor, float]:
    """DDIM sampling"""
    batch_size, _, height, width = conditions.shape
    
    # Initial noise
    image = torch.randn((batch_size, 1, height, width), device=device)
    
    # Set inference steps
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Apply initial noise sigma
    if hasattr(scheduler, 'init_noise_sigma'):
        image = image * scheduler.init_noise_sigma
    
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            
            if hasattr(scheduler, 'scale_model_input'):
                scaled_image = scheduler.scale_model_input(image, timestep)
            else:
                scaled_image = image
            
            
            model_input = torch.cat([conditions, scaled_image], dim=1)
            
            
            timestep_batch = torch.tensor([timestep] * batch_size, device=device).long()
            out = model(model_input, timestep_batch)
            noise_pred = out[0] if isinstance(out, tuple) else out
            
            
            image = scheduler.step(
                noise_pred, timestep, image, eta=ddim_eta,
                use_clipped_model_output=False,
                return_dict=False
            )[0]
    
    end_time = time.time()
    sampling_time = end_time - start_time
    
    return image, sampling_time


def preprocess_conditions(conditions: torch.Tensor) -> torch.Tensor:
    """Preprocess condition inputs, consistent with training"""
    if conditions.size(1) >= 2:
        conditions[:, 0, ...] = conditions[:, 0, ...] + 10.0 * conditions[:, 1, ...]
    return conditions


# ============================================================================
# Metrics calculation functions
# ============================================================================

def calculate_ssim_batch(generated: torch.Tensor, ground_truth: torch.Tensor) -> List[float]:
    """Calculate batch SSIM"""
    batch_size = generated.shape[0]
    ssim_values = []
    
    # Convert to numpy arrays
    gen_np = generated.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    for i in range(batch_size):
        # Get single sample (1, H, W)
        gen_sample = gen_np[i, 0]  # (H, W)
        gt_sample = gt_np[i, 0]    # (H, W)
        
        # Calculate data range
        data_range = gt_sample.max() - gt_sample.min()
        if data_range == 0:
            data_range = 1.0
        
        # Calculate SSIM
        ssim_val = ssim(gt_sample, gen_sample, data_range=data_range)
        ssim_values.append(ssim_val)
    
    return ssim_values


def calculate_metrics(generated: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, List[float]]:
    """Calculate all metrics"""
    with torch.no_grad():
        # Handle NaN and infinite values
        gen = torch.nan_to_num(generated, nan=0.0, posinf=0.0, neginf=0.0)
        gt = torch.nan_to_num(ground_truth, nan=0.0, posinf=0.0, neginf=0.0)
        
        batch_size = gen.shape[0]
        metrics = {
            'MSE': [],
            'NMSE': [],
            'PSNR': [],
            'SSIM': []
        }
        
        # Calculate SSIM
        ssim_values = calculate_ssim_batch(gen, gt)
        metrics['SSIM'] = ssim_values
        
        # Calculate other metrics per sample
        for i in range(batch_size):
            gen_sample = gen[i:i+1]
            gt_sample = gt[i:i+1]
            
            # MSE
            mse_val = F.mse_loss(gen_sample, gt_sample).item()
            metrics['MSE'].append(mse_val)
            
            # NMSE (Normalized MSE)
            gt_power = torch.mean(gt_sample ** 2).item()
            nmse_val = mse_val / gt_power if gt_power > 0 else float('inf')
            metrics['NMSE'].append(nmse_val)
            
            # PSNR
            data_range = (gt_sample.max() - gt_sample.min()).item()
            if data_range <= 1e-12:
                data_range = 1.0
            psnr_val = 20.0 * np.log10(data_range) - 10.0 * np.log10(mse_val) if mse_val > 0 else float('inf')
            metrics['PSNR'].append(psnr_val)
        
        return metrics


# ============================================================================
# Main utility functions
# ============================================================================

def build_model_config(args: argparse.Namespace, in_ch: int, out_ch: int) -> Dict:
    """Build model configuration"""
    return {
        'image_size': args.image_size,
        'in_ch': in_ch + out_ch,  # condition channels + noisy target channels
        'out_ch': out_ch,  # output denoised target
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


def save_results(metrics_all: Dict[str, List[float]], args: argparse.Namespace, 
                total_time: float, avg_time_per_batch: float):
    """Save results to files"""
    # Calculate average metrics
    avg_metrics = {}
    for metric_name, values in metrics_all.items():
        valid_values = [v for v in values if np.isfinite(v)]
        if valid_values:
            avg_metrics[metric_name] = {
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'count': len(valid_values)
            }
        else:
            avg_metrics[metric_name] = {
                'mean': float('nan'),
                'std': float('nan'),
                'count': 0
            }
    
    # Save detailed results
    results = {
        'config': {
            'scheduler_type': args.scheduler_type,
            'num_inference_steps': args.ddpm_steps if args.scheduler_type == 'ddpm' else args.ddim_steps,
            'ddim_eta': args.ddim_eta if args.scheduler_type == 'ddim' else None,
            'data_name': args.data_name,
            'batch_size': args.batch_size,
            'num_samples': args.num_samples if args.num_samples > 0 else 'all'
        },
        'timing': {
            'total_time_seconds': total_time,
            'avg_time_per_batch_seconds': avg_time_per_batch,
            'total_samples': len(metrics_all['MSE'])
        },
        'metrics': avg_metrics
    }
    
    # Save JSON results
    json_path = os.path.join(args.output_dir, f'results_{args.scheduler_type}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV detailed results
    csv_path = os.path.join(args.output_dir, f'detailed_metrics_{args.scheduler_type}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_ID', 'MSE', 'NMSE', 'PSNR', 'SSIM'])
        for i in range(len(metrics_all['MSE'])):
            writer.writerow([
                i,
                metrics_all['MSE'][i],
                metrics_all['NMSE'][i],
                metrics_all['PSNR'][i],
                metrics_all['SSIM'][i]
            ])
    
    print(f"Results saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")


def print_results(metrics_all: Dict[str, List[float]], scheduler_type: str, 
                 total_time: float, total_samples: int):
    """Print results summary"""
    print("\n" + "="*60)
    print(f"{scheduler_type.upper()} Test Results Summary")
    print("="*60)
    
    for metric_name, values in metrics_all.items():
        valid_values = [v for v in values if np.isfinite(v)]
        if valid_values:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            print(f"{metric_name:6s}: {mean_val:8.6f} ± {std_val:8.6f} (n={len(valid_values)})")
        else:
            print(f"{metric_name:6s}: No valid values")
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Total samples: {total_samples}")
    print(f"Average time per sample: {total_time/total_samples:.3f}s")


# ============================================================================
# Command line argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Full test set inference and evaluation script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument('--scheduler_type', type=str, default='ddim', 
                       choices=['ddpm', 'ddim'], help='Scheduler type')
    parser.add_argument('--data_name', type=str, default='Radio', 
                       choices=['Radio', 'Radio_2', 'Radio_3'], help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='RadioMapSeer dataset root directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Trained model checkpoint file path')
    parser.add_argument('--output_dir', type=str, default='./sample_test_results', 
                       help='Results output directory')
    
    # Inference arguments
    parser.add_argument('--ddpm_steps', type=int, default=1000, help='DDPM inference steps')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM inference steps')
    parser.add_argument('--ddim_eta', type=float, default=1.0, help='DDIM eta parameter')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Training diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear', 
                       choices=['linear', 'cosine'], help='Noise schedule type')
    
    # Data arguments
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Data loading worker processes')
    parser.add_argument('--num_samples', type=int, default=-1, 
                       help='Number of test samples, <=0 means full test set')
    
    # Model arguments
    parser.add_argument('--num_channels', type=int, default=96, help='UNet base channels')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of ResNet blocks per layer')
    parser.add_argument('--attention_resolutions', type=str, default='16', 
                       help='Attention mechanism resolution')
    parser.add_argument('--channel_mult', type=str, default='', help='Channel multiplier settings')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--use_checkpoint', type=bool, default=False, help='Whether to use gradient checkpointing')
    parser.add_argument('--use_scale_shift_norm', type=bool, default=True, 
                       help='Whether to use scale-shift normalization')
    parser.add_argument('--resblock_updown', type=bool, default=False, 
                       help='Whether to use up/downsampling in ResNet blocks')
    parser.add_argument('--use_fp16', type=bool, default=False, help='Whether to use half precision')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_head_channels', type=int, default=-1, help='Channels per attention head')
    parser.add_argument('--num_heads_upsample', type=int, default=-1, help='Attention heads for upsampling')
    parser.add_argument('--use_new_attention_order', type=bool, default=False, 
                       help='Whether to use new attention order')
    
    return parser.parse_args()


# ============================================================================
# Main function
# ============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Scheduler type: {args.scheduler_type.upper()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build data loader
    print("Building data loader...")
    dl, in_ch, out_ch = build_dataloader(
        args.data_name, args.data_dir, args.image_size, 
        args.batch_size, args.workers, phase="test"
    )
    
    # Build model
    print("Building model...")
    model_config = build_model_config(args, in_ch, out_ch)
    model = build_unet_from_config(model_config)
    
    # Load model weights
    print(f"Loading model weights: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    # Create scheduler
    scheduler = create_scheduler(
        args.scheduler_type,
        num_train_timesteps=args.diffusion_steps,
        noise_schedule=args.noise_schedule
    )
    
    # Determine inference steps
    if args.scheduler_type == 'ddpm':
        num_inference_steps = args.ddpm_steps
        print(f"DDPM inference steps: {num_inference_steps}")
    else:
        num_inference_steps = args.ddim_steps
        print(f"DDIM inference steps: {num_inference_steps}, eta: {args.ddim_eta}")
    
    # Start inference testing
    print("Starting inference testing...")
    all_metrics = {'MSE': [], 'NMSE': [], 'PSNR': [], 'SSIM': []}
    total_start_time = time.time()
    batch_times = []
    sample_count = 0
    
    for batch_idx, (inputs, image_gain, names) in enumerate(dl):
        if args.num_samples > 0 and sample_count >= args.num_samples:
            break
        
        print(f"Processing batch {batch_idx + 1}/{len(dl)}")
        
        # Prepare data
        if image_gain.dim() == 3:
            image_gain = image_gain.unsqueeze(1)
        
        conditions = preprocess_conditions(inputs.to(device))
        ground_truth = image_gain.to(device)
        
        # Inference
        if args.scheduler_type == 'ddpm':
            generated, sampling_time = sample_ddpm(
                model, scheduler, conditions, num_inference_steps, device
            )
        else:
            generated, sampling_time = sample_ddim(
                model, scheduler, conditions, num_inference_steps, device, args.ddim_eta
            )
        
        batch_times.append(sampling_time)
        sample_count += conditions.shape[0]
        
        # Calculate metrics
        batch_metrics = calculate_metrics(generated, ground_truth)
        for metric_name, values in batch_metrics.items():
            all_metrics[metric_name].extend(values)
        
        # Print batch results
        avg_psnr = np.mean([v for v in batch_metrics['PSNR'] if np.isfinite(v)])
        avg_ssim = np.mean([v for v in batch_metrics['SSIM'] if np.isfinite(v)])
        print(f"  Batch {batch_idx + 1} - PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}, Time: {sampling_time:.2f}s")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_time_per_batch = np.mean(batch_times)
    
    # Print and save results
    print_results(all_metrics, args.scheduler_type, total_time, sample_count)
    save_results(all_metrics, args, total_time, avg_time_per_batch)
    
    print(f"\nTesting completed! Processed {sample_count} samples")


if __name__ == '__main__':
    main()
