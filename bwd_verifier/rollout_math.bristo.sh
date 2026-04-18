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

seed=1
# Run the evaluation script with default parameters
python rollout_math.py \
    --steps 256 \
    --gen_length 256 \
    --block_length 32 \
    --temperature 0.6 \
    --remasking low_confidence \
    --num_rollouts 8 \
    --max_problems -1 \
    --add_solve_instruction \
    --output_file math500_results.add_records.v${seed}.json \
    --verbose \
    --model_path /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed ${seed}

echo "Evaluation completed!"