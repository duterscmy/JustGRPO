#!/bin/bash
#SBATCH --job-name="probe_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

source ~/.bashrc
conda activate ttrl

model_path="${1:?Usage: sbatch $0 MODEL_PATH [LENGTH] [BLOCK_SIZE]}"
length="${2:-256}"
block="${3:-32}"

clean_path="${model_path%/}"
parent_dir="$(basename "$(dirname "$clean_path")")"
base_name="$(basename "$clean_path")"

# 同时包含实验目录和 checkpoint 名，避免不同 checkpoint 覆盖
target_dir="probe_results/${parent_dir}_${base_name}_len${length}_block${block}"
mkdir -p "$target_dir"

log_path="${target_dir}/eval.log"

torchrun --standalone --nproc_per_node=4 \
  eval_gsm8k_probe.py \
  --model_path "$clean_path" \
  --tokenizer_path "/lus/lfs1aip2/projects/public/u6os/mingyu/models/LLaDA-8B-Instruct" \
  --output_dir "$target_dir" \
  --block_size "$block" \
  --num_questions 100 \
  --num_rollouts 64 \
  --rollout_batch_size 4 \
  --temperature 1.0 \
  --steps "$length" \
  --gen_length "$length" \
  --k_values 1 2 4 8 16 32 64 \
  >"$log_path" 2>&1