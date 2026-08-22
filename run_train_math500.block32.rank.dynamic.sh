#!/bin/bash
#SBATCH --job-name="math500_dyn_norm"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

source ~/.bashrc
conda activate ttrl

block=32
t=0.6
lr=5e-6
max_level=5
grad_accum=8

output_dir="./checkpoints_math500_num_generation8_block${block}_t${t}_lr${lr}_level1_${max_level}_dynamic_norm"

if [[ -n "${RUN_TAG:-}" ]]; then
  output_dir="${output_dir}_${RUN_TAG}"
fi

mkdir -p "$output_dir"

accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_math500.rank.py \
  --seed 1997 \
  --run_dir "$output_dir" \
  --temperature "$t" \
  --lr "$lr" \
  --block_size "$block" \
  --max_level "$max_level" \
  --total_steps 10 \
  --save_every 5 \
  --grad_accum "$grad_accum" \
  --scale_by_grad_accum \
  --dynamic_sampling \
  --dynamic_target_valid_groups "$grad_accum" \
  --dynamic_max_attempts_per_group 32 \
  "$@" >> "${output_dir}.log" 2>&1
