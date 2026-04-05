#!/bin/bash
#SBATCH --job-name=llada_math500
#SBATCH --output=logs/llada_math500_%j.out
#SBATCH --error=logs/llada_math500_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=your_partition_name  # Change to your cluster partition

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the evaluation script with default parameters
python rollout_math.py \
    --steps 256 \
    --gen_length 256 \
    --block_length 32 \
    --temperature 0.6 \
    --remasking low_confidence \
    --num_rollouts 2 \
    --max_problems 5 \
    --add_solve_instruction \
    --output_file math500_results.json \
    --save_every 10 \
    --verbose \
    --model_path /mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed 42

echo "Evaluation completed!"