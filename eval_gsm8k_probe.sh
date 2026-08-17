#!/bin/bash
#SBATCH --job-name="probe_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                # 请求1块GPU
#SBATCH --time=10:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名   
conda activate ttrl

model_path=$1
length=${2:-256}
block=${3:-32}
mkdir -p probe_results

# 1. 规范化路径（去除末尾斜杠）
clean_path=$(echo $model_path | sed 's:/*$::')
parent_dir=$(basename $(dirname "$clean_path"))
base_name=$(basename "$clean_path")
target_dir="probe_results/${parent_dir}_${length}_${block}"
mkdir -p "$target_dir"
log_path="${target_dir}/eval.log"

torchrun --standalone --nproc_per_node=4 \
  eval_gsm8k_probe.py \
  --model_path $model_path \
  --tokenizer_path /lus/lfs1aip2/projects/public/u6os/mingyu/models/LLaDA-8B-Instruct \
  --output_dir $target_dir \
  --block_size $block  \
  --num_questions 100 \
  --num_rollouts 64 \
  --rollout_batch_size 4 \
  --temperature 1.0 \
  --steps $length \
  --gen_length $length \
  --k_values 1 2 4 8 16 32 64 &> "$log_path"