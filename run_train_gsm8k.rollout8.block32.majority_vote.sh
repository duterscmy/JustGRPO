#!/bin/bash
#SBATCH --job-name="gsm8k_block32_majority"
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

output_dir="./checkpoints_gsm8k_num_generation8_test_block${block}_temperature${t}_lr${lr}_0828_majority_vote_nodynamic"

# Never append a new run to an existing experiment directory or log.

mkdir -p "$output_dir"

# Uniform majority voting; rollout confidence is logged only.
# Dynamic Sampling is intentionally not enabled.
accelerate launch \
  --num_processes 4 \
  --main_process_ip localhost \
  --config_file configs/fsdp.yaml \
  train_gsm8k.rollout8.majority_vote.py \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6os/mingyu/justgrpo/checkpoints_gsm8k_num_generation8_test_block32_temperature0.6_lr5e-6_0828_majority_vote_nodynamic/training-state-000060 \
  --seed 1997 \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --lr "$lr" \
  --temperature "$t" \
  --total_steps 80 \
  --save_every 5 \
  --grad_accum 8 \
  --scale_by_grad_accum \
  "$@" >> "${output_dir}.log" 2>&1
