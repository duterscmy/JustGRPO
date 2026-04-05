#!/bin/bash
#SBATCH --job-name=llada_gsm8k_rollout
#SBATCH --output=logs/llada_gsm8k_rollout_%j.out
#SBATCH --error=logs/llada_gsm8k_rollout_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=a100

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the evaluation script with default parameters
python rollout_gsm8k.py \
    --steps 256 \
    --gen_length 256 \
    --block_length 32 \
    --temperature 0.6 \
    --remasking low_confidence \
    --num_rollouts 8 \
    --max_problems -1 \
    --add_solve_instruction \
    --output_file gsm8k_results.json \
    --verbose \
    --model_path /mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed 42

echo "Evaluation completed!"