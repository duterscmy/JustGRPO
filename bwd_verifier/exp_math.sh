#!/bin/bash
#SBATCH --job-name=llada_math500_exp
#SBATCH --output=logs/llada_math500_exp_%j.out
#SBATCH --error=logs/llada_math500_exp_%j.err
#SBATCH --time=3:00:00
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

 python exp.py math500_results.add_records.add_digits_bwd.json  \
       math500_results.add_records.add_digits_bwd.eval.fobar.arith.json  \
        --strategies 'fobar' \
        -c arithmetic 