#!/bin/bash
#SBATCH --job-name="rollout_math"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

seed=113
block=$1
t=$2
device=0

export CUDA_VISIBLE_DEVICES=0

# Run the evaluation script with default parameters
python rollout_gsm8k.py \
    --steps 256 \
    --gen_length 256 \
    --block_length ${block} \
    --temperature ${t} \
    --remasking low_confidence \
    --num_rollouts 8 \
    --max_problems -1019 \
    --output_file gsm8k_results.add_records.block${block}.temp${t}.v${seed}.json \
    --verbose \
    --model_path /gpfs/home5/xiaoq/caomingyu/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed ${seed} &> logs/rollout_gsm8k_block${block}_temp${t}_v${seed}.log

echo "Evaluation completed!"