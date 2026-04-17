#!/bin/bash
#SBATCH --job-name="train_arc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # 请求8块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

output_dir=./checkpoints_arc_num_generation8_test_block32
mkdir -p $output_dir

#--resume_ckpt 

accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train_arc.py \
  --run_dir $output_dir \
  --block_size 32 \
  --grad_accum 1 \
  --temperature 1.0 #>> $output_dir.log 2>&1