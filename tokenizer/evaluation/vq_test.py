#!/usr/bin/env python3

import os
import sys
import json
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from einops import rearrange
from skimage.metrics import structural_similarity as ssim
import tensorflow.compat.v1 as tf
# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from tokenizer.dinotok.model_zoo import DINOTOK_HF
from tokenizer.dinotok.ViT_DINO_reg import vit_base_reg
from tokenizer.dinotok.ViT_DINOv3 import vit_base
from tokenizer.modules.lpips import LPIPS
from tokenizer.evaluation.evaluator import Evaluator

# PyTorch Settings for TF32
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def center_crop_arr(
    pil_image: Image.Image,
    image_width: int,
    image_height: int,
) -> Image.Image:
    """
    Center cropping implementation adapted from ADM,
    supporting non-square (image_width, image_height).
    """

    # Step 1: progressively downscale (ADM trick, for antialiasing)
    while (
        pil_image.size[0] >= 2 * image_width
        and pil_image.size[1] >= 2 * image_height
    ):
        pil_image = pil_image.resize(
            (pil_image.size[0] // 2, pil_image.size[1] // 2),
            resample=Image.BOX,
        )

    # Step 2: scale so that both dimensions are >= target
    scale = max(
        image_width / pil_image.size[0],
        image_height / pil_image.size[1],
    )

    pil_image = pil_image.resize(
        (
            round(pil_image.size[0] * scale),
            round(pil_image.size[1] * scale),
        ),
        resample=Image.BICUBIC,
    )

    # Step 3: center crop
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_height) // 2
    crop_x = (arr.shape[1] - image_width) // 2

    return Image.fromarray(
        arr[
            crop_y : crop_y + image_height,
            crop_x : crop_x + image_width,
        ]
    )

def setup_ddp():
    """Initialize DDP environment from torchrun environment variables"""
    if not dist.is_available():
        raise RuntimeError("Distributed training is not available")
    
    # torchrun sets these environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size

def cleanup_ddp():
    """Clean up DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DDP evaluation for image reconstruction")
    
    # Model configuration
    parser.add_argument("--vq-model", type=str, default="dinov3-f16c768-in-1.0")
    parser.add_argument("--vq-ckpt", type=str, 
                       default='/your/dinotok.pt')
    parser.add_argument("--dino-ckpt", type=str,
                       default='/dinov3/pt')
    
    parser.add_argument("--dinov3-status", type=str,
                       default='/dinov3-status/npy')
    
    parser.add_argument("--model-type", type=str, 
                       choices=['v2-b', 'v3-b', 'qwen-image', 'hunyuan_world_vae', 
                               'hunyuan_vae', 'wan_vae', 'sd_vae', 'flux_vae'],
                       default='v3-b', help="Model type to evaluate")
    
    # Model parameters
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--enable-vq", action="store_true")
    parser.add_argument("--codebook-groups-rec", type=int, default=1)
    
    # Image configuration
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=256)
    
    # Data configuration
    parser.add_argument("--data-path", type=str,
                       default='/imagenet/origin/val',
                       help="Path to image tensor file (.pt)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    
    parser.add_argument("--save-gt", action="store_true", help="Save ground truth images")
    parser.add_argument("--save-fake", action="store_true", help="Save reconstructed images")
    parser.add_argument("--output-dir", type=str, default="/job_data/outputs")
    parser.add_argument("--fake-dir", type=str, default=None)
    parser.add_argument("--gt-dir", type=str, default=None)

    parser.add_argument("--ref-batch", default="/public_data/imagenet/origin/gt_256.npz", help="path to reference batch npz file")
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()
        # Set default directories if not specified
    if args.fake_dir is None:
        args.fake_dir = os.path.join(args.output_dir, args.model_type, "fakedir")
        os.makedirs(args.fake_dir, exist_ok=True)
    if args.gt_dir is None:
        args.gt_dir = os.path.join(args.output_dir, args.model_type, "gtdir")
        os.makedirs(args.gt_dir, exist_ok=True)
    return args



def load_models(args, device, local_rank):

    if args.model_type in ["v2-b", "v3-b"]:
        # DINO-based models
        tokenizer = DINOTOK_HF(args).to(device).eval()
        ckpt = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(ckpt["model"], strict=False)

        if args.model_type == 'v2-b':
            dino = vit_base_reg(checkpoint=args.dino_ckpt).to(device).eval()
        elif args.model_type == 'v3-b':
            dino = vit_base(checkpoint=args.dino_ckpt).to(device).eval()
        
        del ckpt
        return tokenizer, dino
   
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    

def get_model_decoder_image(im_t, model, model_type, pc_sortids_top_tensor):
    """Decode image tensor through the model"""
    _, _, hh, ww = im_t.shape
    q_indices = None
    if model_type == "v2-b" and isinstance(model, tuple):
        tokenizer, dino = model
        dino_feat = dino(im_t)['x_norm_patchtokens']
        dino_feat = rearrange(dino_feat, 'b (h w) c -> b c h w', h=hh//14, w=ww//14)
        image, _, _ = tokenizer(dino_feat)
    
    elif model_type == "v3-b" and isinstance(model, tuple):
        tokenizer, dino = model
        dino_feat = dino.get_intermediate_layers(im_t, reshape=False, norm=True)
        dino_feat = tuple(rearrange(feat, 'b (h w) c -> b c h w', h=hh//16, w=ww//16) for feat in dino_feat)
        image, _, vq_feat,_, q_indices = tokenizer(dino_feat, None)

    elif model_type in ["sd_vae", "flux_vae"]:
        vae = model
        posterior = vae.encode(im_t).latent_dist
        latent = posterior.sample()
        image = vae.decode(latent).sample
    
    elif model_type in ["qwen-image", "hunyuan_world_vae", "hunyuan_vae"]:
        vae = model
        im_5d = im_t.unsqueeze(2)  # [B, C, 1, H, W]
        posterior = vae.encode(im_5d).latent_dist
        latent = posterior.sample()
        rec_5d = vae.decode(latent).sample
        image = rec_5d.squeeze(2)
    
    elif model_type == "wan_vae":
        vae = model
        outs = []
        for b in range(im_t.shape[0]):
            seq_4d = im_t[b].unsqueeze(1).to(getattr(vae, "device", im_t.device))
            lat = vae.encode([seq_4d])[0]
            rec = vae.decode([lat])[0].squeeze(1).cpu()
            outs.append(rec)
        image = torch.stack(outs, dim=0).to(im_t.device)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return image,q_indices

def get_normalization_params(model_type):
    """Get normalization parameters for different model types"""
    if model_type == 'v3-b':
        inmean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        instd = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    else:
        inmean = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
        instd = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
    return inmean, instd

def evaluate_batch(images_gt, images_rec, lpips_model, device):
    """Compute metrics for a batch of images"""
    batch_metrics = {
        'mse': 0.0,
        'psnr': 0.0,
        'ssim': 0.0,
        'lpips': 0.0,
        'count': 0
    }
    
    # LPIPS for the whole batch
    lpips_val = lpips_model(images_rec, images_gt).sum().item()
    batch_metrics['lpips'] += lpips_val
    
    for i in range(images_gt.shape[0]):
        # MSE and PSNR
        mse = F.mse_loss(images_rec[i], images_gt[i]).item()
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
        
        # SSIM
        img_u8 = (images_rec[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        gt_u8 = (images_gt[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        ssim_val = ssim(img_u8, gt_u8, channel_axis=-1)
        
        batch_metrics['mse'] += mse
        batch_metrics['psnr'] += psnr
        batch_metrics['ssim'] += ssim_val
        batch_metrics['count'] += 1
    
    return batch_metrics

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not is_dist_avail_and_initialized():
        return tensor.detach()
    tensors_gather = [torch.zeros_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def main():
    """Main evaluation function"""
    # Setup DDP
    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"
    
    # Parse arguments
    args = parse_args()
    
    if rank == 0:
        print(f"=" * 80)
        print(f"Starting evaluation with {world_size} GPUs")
        print(f"Model type: {args.model_type}")
        print(f"Batch size: {args.batch_size}")
        print(f"=" * 80)
    
    if world_size > 1:
        dist.barrier()  # Ensure directories are created
    
    # Load models
    model = load_models(args, device, local_rank)
    
    # Load LPIPS
    lpips_model = LPIPS().to(device)
    
    # Load image tensor
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_width, args.image_height)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    
    # Get normalization parameters
    inmean, instd = get_normalization_params(args.model_type)
    inmean, instd = inmean.to(device), instd.to(device)

    # Distribute data across ranks
    indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    
    if rank == 0:
        print(f"Each GPU processes {len(subset)} images")
    
    # Initialize metrics
    total_metrics = {
        'mse': torch.tensor(0.0, device=device),
        'psnr': torch.tensor(0.0, device=device),
        'ssim': torch.tensor(0.0, device=device),
        'lpips': torch.tensor(0.0, device=device),
        'count': torch.tensor(0.0, device=device)
    }
    
    iterator = enumerate(loader)
    if rank==0:
        iterator = tqdm(iterator, total=len(loader)) 

    val_imgs = []

    # slot_indices1 = torch.Tensor([]).to(device)
    # slot_indices2 = torch.Tensor([]).to(device)

    for batch_idx, (im_t, _) in iterator:
        im_t = im_t.to(device).float()
        im_t =  (im_t - inmean) / instd
        with torch.no_grad():

            image_rec, q_indices = get_model_decoder_image(im_t, model, args.model_type, None)

            # slot_indices1 = torch.unique(torch.cat((slot_indices1, concat_all_gather(q_indices[0]))))
            # slot_indices2 = torch.unique(torch.cat((slot_indices2, concat_all_gather(q_indices[1]))))

            image_vis_local = (image_rec * instd + inmean).clamp(0, 1)
            gt_vis_local = (im_t * instd + inmean).clamp(0, 1)
            
            # Compute metrics
            batch_metrics = evaluate_batch(gt_vis_local, image_vis_local, lpips_model, device)
            
            # Accumulate metrics
            total_metrics['mse'] += batch_metrics['mse']
            total_metrics['psnr'] += batch_metrics['psnr']
            total_metrics['ssim'] += batch_metrics['ssim']
            total_metrics['lpips'] += batch_metrics['lpips']
            total_metrics['count'] += batch_metrics['count']
            
            
            # save npz in memory
            image_vis = concat_all_gather(image_vis_local)
            gt_vis = concat_all_gather(gt_vis_local)
            image_vis_u8 = image_vis * 255.
            image_vis_u8 = image_vis_u8.permute(0,2,3,1).to('cpu', dtype=torch.uint8).numpy()
            val_imgs.append(image_vis_u8)  
            
            # Save images if requested
            if args.save_fake or args.save_gt:
                for i in range(image_rec.shape[0]):
                    idx = indices[batch_idx * args.batch_size + i]

                    if args.save_fake:
                        img = (image_vis_local[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                        Image.fromarray(img).save(os.path.join(args.fake_dir, f"{idx}_r{rank}.png"))

                    if args.save_gt:
                        img = (gt_vis_local[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                        Image.fromarray(img).save(os.path.join(args.gt_dir, f"{idx}_r{rank}.png"))
    
    # Aggregate metrics across all GPUs
    if world_size > 1:
        dist.barrier()
        for key in total_metrics:
            dist.all_reduce(total_metrics[key], op=dist.ReduceOp.SUM)

    
    # Compute final metrics (only on rank 0)
    if rank == 0:

        # num_slots = 16384
        # usage_slot1 = slot_indices1.size(0) / num_slots
        # usage_slot2 = slot_indices2.size(0) / num_slots

        count = total_metrics['count'].item()
        mse_avg = total_metrics['mse'].item() / count
        psnr_avg = total_metrics['psnr'].item() / count
        ssim_avg = total_metrics['ssim'].item() / count
        lpips_avg = total_metrics['lpips'].item() / count
        psnr_from_mse = 20 * math.log10(1.0 / math.sqrt(mse_avg))
        val_imgs = np.concatenate(val_imgs, axis=0)
        print(f'validation samples: {val_imgs.shape[0]}')
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        evaluator = Evaluator(tf.Session(config=config),batch_size=64)
        evaluator.warmup()
        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(args.ref_batch)
        print("computing/reading reference batch statistics...")
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)
        
        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(val_imgs)
        print("computing/reading sample batch statistics...")
        sample_stats, _ = evaluator.read_statistics(val_imgs, sample_acts)
        FID, FIDmu, FIDsigma = sample_stats.frechet_distance(ref_stats)

        IS = evaluator.compute_inception_score(sample_acts[0])

        print("Computing evaluations...")
        print("Inception Score:", IS)
        print("rFID:", FID)
        print("rFIDmu:", FIDmu)
        print("rFIDsigma:", FIDsigma)
       #  print(f"Codebook usage - Slot1: {usage_slot1:.4f}, Slot2: {usage_slot2:.4f}")
        print("\n" + "=" * 80)
        print("FINAL EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model Type:        {args.model_type}")
        print(f"Total Images:      {int(count)}")
        print(f"MSE:               {mse_avg:.6f}")
        print(f"PSNR (averaged):   {psnr_avg:.4f} dB")
        print(f"PSNR (from MSE):   {psnr_from_mse:.4f} dB")
        print(f"SSIM:              {ssim_avg:.6f}")
        print(f"LPIPS:             {lpips_avg:.6f}")
        print("=" * 80)
        print(f"Note: PSNR difference = {10000 * abs(psnr_avg - psnr_from_mse):.4f} dB")
        
        # Save results to JSON
        results = {
            "model_type": args.model_type,
            "count": int(count),
            "mse": mse_avg,
            "psnr": psnr_avg,
            "psnr_from_mse": psnr_from_mse,
            "ssim": ssim_avg,
            "lpips": lpips_avg,
            "FID": FID.item(),
            "IS": IS,
            "vq_ckpt": args.vq_ckpt if args.model_type in ["v2-b", "v3-b"] else "N/A",
            "dino_ckpt": args.dino_ckpt if args.model_type in ["v2-b", "v3-b"] else "N/A"
        }
        
        result_path = os.path.join(args.output_dir, f"{args.model_type}_results.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {result_path}\n")
    
    # Cleanup
    if world_size > 1:
        dist.barrier()
        cleanup_ddp()

if __name__ == "__main__":
    main()
