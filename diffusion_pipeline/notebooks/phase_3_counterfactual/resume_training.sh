#!/bin/bash
# Resume training from Stage 1 checkpoint

cd /scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-gpu

# Set GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -k2 -nr | head -1 | cut -d' ' -f1)

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Resuming from Stage 1 checkpoint (epoch 50)..."

# Resume training with --resume-stage1 flag
nohup python 16_enhanced_diffusion_cf.py --resume-stage1 > enhanced_diffusion_stage2.log 2>&1 &

echo "Stage 2 training started! Check log:"
echo "tail -f enhanced_diffusion_stage2.log"
