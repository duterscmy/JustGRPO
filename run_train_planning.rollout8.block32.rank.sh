#!/bin/bash
#SBATCH --job-name="train_planning"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -euo pipefail

source ~/.bashrc
conda activate opd

task=${1:?"Usage: sbatch $0 <sudoku|countdown> [model_path] [data_dir]"}
model_path=${2:-${MODEL_PATH:-/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct}}
data_dir=${3:-${DATA_DIR:-dataset}}

case "$task" in
  sudoku)
    temperature=${TEMPERATURE:-0.3}
    ;;
  countdown)
    temperature=${TEMPERATURE:-1.0}
    ;;
  *)
    echo "task must be sudoku or countdown" >&2
    exit 2
    ;;
esac

block=${BLOCK_SIZE:-32}
lr=${LEARNING_RATE:-5e-6}
total_steps=${TOTAL_STEPS:-30}
output_dir=${OUTPUT_DIR:-./checkpoints_${task}_rollout8_block${block}_temperature${temperature}_lr${lr}}
mkdir -p "$output_dir"

accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_planning.rollout8.rank.py \
  --task "$task" \
  --data_dir "$data_dir" \
  --model_path "$model_path" \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --lr "$lr" \
  --temperature "$temperature" \
  --total_steps "$total_steps" \
  --save_every 5 \
  --grad_accum 8 \
  --num_generations 4 \
  --sample_repeat_times 2 \
  --gen_length 256 \
  --gen_steps 256 \
  > "${output_dir}.log" 2>&1
