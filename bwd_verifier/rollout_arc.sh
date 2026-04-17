#!/bin/bash
#SBATCH --job-name=llada_arc_rollout
#SBATCH --output=logs/llada_arc_rollout_%j.out
#SBATCH --error=logs/llada_arc_rollout_%j.err
#SBATCH --time=24:00:00
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
python rollout_arc.py \
    --steps 64 \
    --gen_length 64 \
    --block_length 8 \
    --temperature 1.0 \
    --remasking low_confidence \
    --num_rollouts 8 \
    --max_problems -1 \
    --add_solve_instruction \
    --output_file arc_results.add_records.json \
    --verbose \
    --model_path /mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed 42

echo "Evaluation completed!"