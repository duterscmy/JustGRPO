#!/bin/bash
#SBATCH --job-name="gsm8k_block32_adaptive_t"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -e

source ~/.bashrc
conda activate ttrl

block=32
initial_temperature=1.0
lr=5e-6

output_dir="./checkpoints_gsm8k_num_generation8_test_block${block}_temperature${initial_temperature}_lr${lr}_0826_adaptive_temperature_uniform_nodynamic"

if [[ -e "$output_dir" || -e "${output_dir}.log" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  echo "Set RUN_TAG to a new value for another run." >&2
  exit 1
fi

mkdir -p "$output_dir"

# The first optimizer step calibrates the confidence target.  Afterwards the
# controller updates once per optimizer step because its default window equals
# --grad_accum.  Uniform voting remains unchanged and Dynamic Sampling is off.
accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_gsm8k.rollout8.adaptive_temperature.py \
  --seed 1997 \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --lr "$lr" \
  --temperature "$initial_temperature" \
  --total_steps 15 \
  --save_every 5 \
  --grad_accum 8 \
  --scale_by_grad_accum \
  "$@" >> "${output_dir}.log" 2>&1
