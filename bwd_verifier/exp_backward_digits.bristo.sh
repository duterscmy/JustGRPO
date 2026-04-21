#!/bin/bash
#SBATCH --job-name="bwd_verify"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/bwd_verify_%j.out
#SBATCH --error=logs/bwd_verify_%j.err

# ### 激活conda环境
# source ~/.bashrc # 你的环境名
# conda activate ttrl

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

version=1
input=$1
output=$2ß
python exp_backward_digits.py $input $output \
    --verbose --only_result -m /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct