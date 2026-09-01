#!/bin/bash
#SBATCH --job-name="gsm8k_ar_dynamic"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -eo pipefail

source ~/.bashrc
conda activate ttrl
set -u

block=1
initial_temperature="${INITIAL_TEMPERATURE:-0.6}"
lr="${LR:-5e-6}"
total_steps="${TOTAL_STEPS:-80}"
save_every="${SAVE_EVERY:-5}"
grad_accum="${GRAD_ACCUM:-8}"
seed="${SEED:-1997}"
max_attempts="${DYNAMIC_MAX_ATTEMPTS:-32}"

run_tag="${RUN_TAG:-seed${seed}}"
output_dir="./checkpoints_gsm8k_rollout8_block${block}_temperature${initial_temperature}_lr${lr}_dynamic_sampling_${run_tag}"
log_path="${output_dir}.log"

if [[ -e "$output_dir" || -e "$log_path" ]]; then
  echo "Refusing to overwrite an existing run: $output_dir" >&2
  echo "Set RUN_TAG to a new value for another run." >&2
  exit 1
fi

mkdir -p "$output_dir"

model_args=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  model_args+=(--model_path "$MODEL_PATH")
fi

accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_gsm8k.rollout8.majority_vote.dynamic_sampling.py \
  --seed "$seed" \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --lr "$lr" \
  --temperature "$initial_temperature" \
  --total_steps "$total_steps" \
  --save_every "$save_every" \
  --grad_accum "$grad_accum" \
  --scale_by_grad_accum \
  --dynamic_sampling \
  --dynamic_target_valid_groups "$grad_accum" \
  --dynamic_max_attempts_per_group "$max_attempts" \
  "${model_args[@]}" \
  "$@" >> "$log_path" 2>&1
