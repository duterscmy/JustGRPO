#!/bin/bash
#SBATCH --job-name="gsm8k_ar_divtemp"
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
initial_temperature="${INITIAL_TEMPERATURE:-1.0}"
lr="${LR:-5e-6}"
save_every="${SAVE_EVERY:-5}"
grad_accum="${GRAD_ACCUM:-8}"
seed="${SEED:-1997}"

target_diversity="${TARGET_DIVERSITY:-2.0}"
ema_decay="${DIVERSITY_EMA_DECAY:-0.6}"
controller_gain="${DIVERSITY_GAIN:-0.5}"
deadband="${DIVERSITY_DEADBAND:-0.1}"
max_change="${DIVERSITY_MAX_CHANGE:-0.10}"
min_temperature="${DIVERSITY_TEMP_MIN:-0.3}"
max_temperature="${DIVERSITY_TEMP_MAX:-1.5}"

run_tag="${RUN_TAG:-seed${seed}}"
output_dir="./checkpoints_gsm8k_rollout8_block${block}_temperature${initial_temperature}_lr${lr}_diversity_target${target_diversity}_${run_tag}"
log_path="${output_dir}.log"


mkdir -p "$output_dir"

model_args=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  model_args+=(--model_path "$MODEL_PATH")
fi

accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_gsm8k.rollout8.diversity_temperature.py \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6os/mingyu/justgrpo/checkpoints_gsm8k_rollout8_block1_temperature1.0_lr5e-6_diversity_target2.0_seed1997/training-state-000075 \
  --seed "$seed" \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --lr "$lr" \
  --temperature "$initial_temperature" \
  --total_steps 100 \
  --save_every "$save_every" \
  --grad_accum "$grad_accum" \
  --scale_by_grad_accum \
  --diversity_temp_target "$target_diversity" \
  --diversity_temp_ema_decay "$ema_decay" \
  --diversity_temp_gain "$controller_gain" \
  --diversity_temp_deadband "$deadband" \
  --diversity_temp_max_change "$max_change" \
  --diversity_temp_min "$min_temperature" \
  --diversity_temp_max "$max_temperature" \
  "${model_args[@]}" \
  "$@" >> "$log_path" 2>&1
