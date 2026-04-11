# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, os, pdb
import os.path as osp
import math, torch, time
import torch.distributed as dist
from omegaconf import OmegaConf
import torch.nn.functional as F
import torch.distributed as dist
import tensorflow.compat.v1 as tf
from tokenizer.evaluation.evaluator import Evaluator
torch.backends.cudnn.allow_tf32 = True
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from tokenizer.dinotok.model_zoo import DINOTOK_HF
from tokenizer.dinotok.ViT_DINO_reg import vit_base_reg
from tokenizer.dinotok.ViT_DINOv3 import vit_base
torch.backends.cuda.matmul.allow_tf32 = True
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

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

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    
    #* Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    #* Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup vq model
    vq_model = DINOTOK_HF(args)
    print(float(sum(p.numel() for p in vq_model.parameters()))/1024/1024)
    vq_model = vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(checkpoint["model"], strict=True)
    if args.dino_version == 'v2-b':
        dino_model = vit_base_reg(checkpoint=args.dino_ckpt).cuda()
    elif args.dino_version == 'v3-b':
        dino_model = vit_base(checkpoint=args.dino_ckpt).cuda() 
    dino_model.eval()
    del checkpoint

    #* Create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu",weights_only=False)

    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif "ema" in checkpoint:  # ddp
        model_weight = checkpoint['ema']
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")

    m1, u1 = gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True,
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    ckpt_string_name = args.gpt_ckpt.split("/")[-1].split("_")[-1].split(".")[0]
    # time_record = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    folder_name = f"{model_string_name}-arhead-{ckpt_string_name}-size-{args.image_size}-" \
                  f"topk-{args.top_k_first}-topp-{args.top_p_first}-T-{args.temperature}-"\
                  f"cfg-{args.cfg_scale_first}-{args.cfg_scale_last}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    world_size = dist.get_world_size()
    global_batch_size = n * world_size
    
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
   
    assert total_samples % world_size == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    gen_samples = []
    total = 0
    for _ in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        qzshape = [len(c_indices), latent_size, latent_size, args.codebook_embed_dim]
        index_sample = generate( 
            gpt_model, c_indices, latent_size ** 2,
            cfg_scale_first=args.cfg_scale_first,cfg_scale_last=args.cfg_scale_last, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k_first=args.top_k_first,top_k_last=args.top_k_last,
            top_p_first=args.top_p_first,  top_p_last=args.top_p_last, sample_logits=True, 
            )
        
        emb_first_norm = F.normalize(vq_model.quantize.embedding_first.weight.data.to(device=device, dtype=precision), p=2, dim=-1)  # [b, l, V] * [V, d] -> [b, l ,d]
        emb_last_norm = F.normalize(vq_model.quantize.embedding_last.weight.data.to(device=device, dtype=precision), p=2, dim=-1)  # [b, l, V] * [V, d] -> [b, l ,d]
        emb_first_sample = emb_first_norm[index_sample[:,:,0]]
        emb_last_sample = emb_last_norm[index_sample[:,:,1]]

        emb_sample = torch.cat((emb_last_sample,emb_first_sample), dim=-1)
        samples,_ = vq_model.decode(emb_sample.contiguous().view(qzshape).permute(0,3,1,2))
        inmean=torch.tensor((0.485, 0.456, 0.406), device=device)
        instd=torch.tensor((0.229, 0.224, 0.225), device=device)
        samples = samples * instd[:, None, None] + inmean[:, None, None]
        samples = torch.clamp(255.0 * samples, 0, 255).permute(0, 2, 3, 1).contiguous()

    
        samples = concat_all_gather(samples)
        samples = samples.to('cpu', dtype=torch.uint8).numpy()
        gen_samples.append(samples)  

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:

        gen_samples = np.concatenate(gen_samples, axis=0)[:50_000]
        print(f'generated samples: {gen_samples.shape[0]}')

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
        sample_acts = evaluator.read_activations(gen_samples)
        print("computing/reading sample batch statistics...")
        sample_stats, sample_stats_spatial = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)
        sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)

        IS = evaluator.compute_inception_score(sample_acts[0])
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

        print("Computing evaluations...")
        print("Inception Score:", IS)
        print("FID:", FID)
        print("sFID:", sFID)
        print("Precision:", prec)
        print("Recall:", recall)

        print("Done.")
    
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--dino-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='none', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, default=None)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=832, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256,336, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale-first",  type=float, default=1.5)
    parser.add_argument("--cfg-scale-last",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--dino-version", type=str, choices=['v2-b', 'v3-b'], default='v3-b', help="dino version to use")
    parser.add_argument("--top-k-first", type=int, default=4096,help="top-k value to sample with")
    parser.add_argument("--top-k-last", type=int, default=16384,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p-first", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--top-p-last", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--image-height", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--image-width", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--z-channels", type=int, default=512,)
    parser.add_argument("--ref-batch", type=str, default=None, help="path to reference batch npz file")
    parser.add_argument("--enable-vq", action='store_true', help='use vq')
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--codebook_groups_rec", type=int, default=2) 
    parser.add_argument("--dinov3-status", type=str,
                    default='/dinov3-status/npy')
    
    args = parser.parse_args()
    main(args)