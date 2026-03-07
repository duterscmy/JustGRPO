#!/bin/bash
#SBATCH --job-name="train_math_16"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # 请求8块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl


output_dir=./checkpoints_math500_num_generation16
mkdir -p $output_dir
accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train_math500.py \
  --run_dir $output_dir \
  --grad_accum 16 #>> $output_dir.log 2>&1