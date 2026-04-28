#!/bin/bash
#SBATCH --job-name="rollout_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
seed=42
block=$1
t=$2
# Run the evaluation script with default parameters
python rollout_gsm8k.py \
    --steps 256 \
    --gen_length 256 \
    --block_length ${block} \
    --temperature ${t} \
    --remasking low_confidence \
    --num_rollouts 8 \
    --max_problems 300 \
    --output_file gsm8k_results.300.add_records.${block}.${t}.seed${seed}.json \
    --verbose \
    --model_path /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed ${seed}

echo "Evaluation completed!"