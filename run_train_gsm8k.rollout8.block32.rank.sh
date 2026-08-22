#!/bin/bash
#SBATCH --job-name="train_gsm8k_block32"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

block=32
t=1.0
lr=5e-6
output_dir=./checkpoints_gsm8k_num_generation8_test_block${block}_temperature${t}_lr${lr}_0814_rank

# 同一个脚本跑多组配置时可用 RUN_TAG 隔离输出，例如：
# RUN_TAG=norm / no_norm / dynamic
if [[ -n "${RUN_TAG:-}" ]]; then
  output_dir="${output_dir}_${RUN_TAG}"
fi

mkdir -p "$output_dir"

#--resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.rollout8.rank.py \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6os/mingyu/justgrpo/checkpoints_gsm8k_num_generation8_test_block32_temperature1.0_lr5e-6_0814_rank/training-state-000025 \
  --seed 1997 \
  --run_dir "$output_dir" \
  --block_size "$block" \
  --lr "$lr" \
  --temperature "$t" \
  --total_steps 45 --save_every 5 \
  --grad_accum 8 \
  "$@" >> "${output_dir}.log" 2>&1
