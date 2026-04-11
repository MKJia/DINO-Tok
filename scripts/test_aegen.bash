# !/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    lightningdit/sample_ddp_dinotok.py \
    --precision fp32 \
    --config lightningdit/configs/DiTDHXL-DINOv3-AE.yaml \
    --sample-dir /path/to/sample \
    --ref-batch /path/to/VIRTUAL_imagenet256_labeled.npz \
    --ckpt-path "/path/to/aegen/ckpt" \
    --step 800 \
    --per-proc-batch-size 8 \
    --stat-path "/path/to/stat" \
    --sample-step 25
        
# bash scripts/test_aegen.bash

