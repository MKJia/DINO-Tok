# !/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"


torchrun --nproc_per_node=4 \
         --nnodes=1 \
         tokenizer/evaluation/ae_test.py \
         --vq-model dinov3-f16c768-in-1.0 \
         --model-type v3-b \
         --dino-ckpt /path/to/dinov3 \
         --ref-batch /path/to/gt_256.npz \
         --batch-size 8 \
         --vq-ckpt /path/to/ae \
         --codebook-size 16384 \
         --data-path /path/to/imagenet/val \
         --output-dir ../tokenizer/evaluation/output_ae

# Usage:
#   bash scripts/test_aetok.bash
