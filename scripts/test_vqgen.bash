# !/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 \
         --nnodes=1 \
         autoregressive/test_c2i.py \
         --image-size 256 \
         --image-height 256 \
         --image-width 256 \
         --vq-model dinov3-f16c768-in-1.0 \
         --vq-ckpt /path/to/vq \
         --sample-dir /path/to/sample \
         --gpt-model GPT-L \
         --ref-batch /path/to/VIRTUAL_imagenet256_labeled.npz \
         --gpt-ckpt /path/to/vqgen/ckpt \
         --cfg-scale-first 3.20 \
         --cfg-scale-last 1.15 \
         --codebook-embed-dim 96 \
         --codebook-size 16384 \
         --enable-vq \
         --per-proc-batch-size 8 \
         --precision none \
         --num-fid-samples 50000 \
         --downsample-size 16 \
         --compile \
         --dinov3-status /path/to/dinov3/dinov3_pca_stat.npy \
         --dino-ckpt /path/to/dinov3

# # "L","XXL"
# torchrun --nproc_per_node=2 \
#          --nnodes=1 \
#          autoregressive/test_c2i.py \
#          --image-size 256 \
#          --image-height 256 \
#          --image-width 256 \
#          --vq-model dinov3-f16c768-in-1.0 \
#          --vq-ckpt /dinotok_vq/pt \
#          --sample-dir /sample_results \
#          --gpt-model GPT-L \
#          --gpt-ckpt /llamagan/pt \
#          --cfg-scale-first 3.20 \
#          --cfg-scale-last 1.15 \
#          --codebook-embed-dim 96 \
#          --codebook-size 16384 \
#          --enable-vq \
#          --per-proc-batch-size 8 \
#          --precision none \
#          --num-fid-samples 50000 \
#          --downsample-size 16 \
#          --compile \
#          --ref-batch /imagenet/origin/gt_256.npz \
#          --dinov3-status /DINOV3/status/npy \
#          --dino-ckpt /dinov3/pth

# bash scripts/test_vqgen.bash

