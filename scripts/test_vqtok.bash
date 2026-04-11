# !/bin/bash
export CUDA_VISIBLE_DEVICES="4,5,6,7" 

torchrun --nproc_per_node=4 \
         --nnodes=1 \
         tokenizer/evaluation/vq_test.py \
         --vq-model dinov3-f16c768-in-1.0 \
         --model-type v3-b \
         --dino-ckpt /path/to/dinov3 \
         --ref-batch /path/to/gt_256.npz \
         --batch-size 8 \
         --vq-ckpt /path/to/vq \
         --codebook-size 16384 \
         --enable-vq \
         --data-path /path/to/imagenet/val \
         --dinov3-status /path/to/dinov3/dinov3_pca_stat.npy \
         --output-dir ../tokenizer/evaluation/output_vq



# Usage:
#   bash scripts/test_vqtok.bash
