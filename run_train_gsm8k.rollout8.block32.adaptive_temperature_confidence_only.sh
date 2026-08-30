#!/bin/bash
#SBATCH --job-name="gsm8k_block32_adaptive_t_v2"
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
initial_temperature="${INITIAL_TEMPERATURE:-1.0}"
lr="${LR:-5e-6}"

output_dir="./checkpoints_gsm8k_num_generation8_test_block${block}_temperature${initial_temperature}_lr${lr}_0828_adaptive_temperature_confidence_only_v2"


mkdir -p "$output_dir"

accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_gsm8k.rollout8.adaptive_temperature_confidence_only.py \
  --resume_ckpt  /lus/lfs1aip2/projects/public/u6os/mingyu/justgrpo/checkpoints_gsm8k_num_generation8_test_block32_temperature1.0_lr5e-6_0828_adaptive_temperature_confidence_only_v2/training-state-000020 \
  --seed 1997 \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --lr "$lr" \
  --temperature "$initial_temperature" \
  --total_steps 50 \
  --save_every 5 \
  --grad_accum 8 \
  --scale_by_grad_accum \
  --adaptive_temp_calibration_windows "${ADAPTIVE_CALIBRATION_WINDOWS:-2}" \
  --adaptive_temp_ema_decay "${ADAPTIVE_EMA_DECAY:-0.5}" \
  --adaptive_temp_confidence_gain "${ADAPTIVE_CONFIDENCE_GAIN:-3.0}" \
  --adaptive_temp_deadband "${ADAPTIVE_DEADBAND:-0.0005}" \
  --adaptive_temp_max_change "${ADAPTIVE_MAX_CHANGE:-0.05}" \
  --adaptive_temp_min "${ADAPTIVE_TEMP_MIN:-0.5}" \
  --adaptive_temp_max "${ADAPTIVE_TEMP_MAX:-1.5}" \
  "$@" >> "${output_dir}.log" 2>&1
