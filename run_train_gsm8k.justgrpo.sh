#!/bin/bash
#SBATCH --job-name="train_gsm8k_just"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # 请求8块GPU
#SBATCH --time=22:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

output_dir=./checkpoints_gsm8k_justgrpo
mkdir -p $output_dir
# --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_gsm8k_justgrpo/training-state-000005
accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.justgrpo.py \
  --run_dir $output_dir \
  --grad_accum 16 \
 --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_gsm8k_justgrpo/training-state-000028/ >> $output_dir.log 2>&1