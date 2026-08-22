#!/bin/bash
#SBATCH --job-name="arc_c_dyn_norm"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

source ~/.bashrc
conda activate ttrl

block=32
t=1.0
lr=1e-6
length=128
grad_accum=16

output_dir="./checkpoints_arc_c_num_generation8_length${length}_block${block}_t${t}_lr${lr}_dynamic_norm"

if [[ -n "${RUN_TAG:-}" ]]; then
  output_dir="${output_dir}_${RUN_TAG}"
fi

mkdir -p "$output_dir"

accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_arc.py \
  --seed 1997 \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --grad_accum "$grad_accum" \
  --lr "$lr" \
  --total_steps 25 \
  --save_every 5 \
  --temperature "$t" \
  --gen_length "$length" \
  --gen_steps "$length" \
  --scale_by_grad_accum \
  --dynamic_sampling \
  --dynamic_target_valid_groups "$grad_accum" \
  --dynamic_max_attempts_per_group 32 \
  "$@" >> "${output_dir}.log" 2>&1
