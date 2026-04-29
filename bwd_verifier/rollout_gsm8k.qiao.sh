#!/bin/bash
#SBATCH --job-name=llada_gsm8k_rollout
#SBATCH --output=logs/llada_gsm8k_rollout_%j.out
#SBATCH --error=logs/llada_gsm8k_rollout_%j.err
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
    --output_file gsm8k_results.add_records.tmp1.0.v${seed}.json \
    --verbose \
    --model_path /gpfs/home5/xiaoq/caomingyu/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed ${seed}

echo "Evaluation completed!"