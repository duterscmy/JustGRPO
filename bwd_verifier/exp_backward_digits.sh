# !/bin/bash
# SBATCH --job-name=bwd_verify
# SBATCH --output=logs/bwd_verify_%j.out
# SBATCH --error=logs/bwd_verify_%j.err
# SBATCH --time=24:00:00
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --gres=gpu:1
# SBATCH --cpus-per-task=8
# SBATCH --mem=64G
# SBATCH --partition=a100

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

input=data/math500_results.add_records.v1.json
output=data/math500_results.add_records.v1.backward_digits.json
python exp_backward_digits.py $input $output --verbose