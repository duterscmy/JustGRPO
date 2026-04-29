#!/bin/bash
#SBATCH --job-name="rollout_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err
#SBATCH --cpus-per-task=18
#SBATCH --mem 64G
#SBATCH --partition=gpu_h100

# Create logs directory if it doesn't exist
mkdir -p logs
source /gpfs/home5/xiaoq/.bashrc
conda activate ttrl
cd /home/xiaoq/caomingyu/justgrpo/bwd_verifier
# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Set environment variables for PyTorch
seed=42
block=$1
t=$2
device=0

export CUDA_VISIBLE_DEVICES=$device

# Run the evaluation script with default parameters
python rollout_gsm8k.py \
    --steps 256 \
    --gen_length 256 \
    --block_length ${block} \
    --temperature ${t} \
    --remasking low_confidence \
    --num_rollouts 8 \
    --max_problems 300 \
    --output_file gsm8k_results.add_records.block${block}.temp${t}.v${seed}.json \
    --verbose \
    --model_path /gpfs/home5/xiaoq/caomingyu/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed ${seed}

echo "Evaluation completed!"