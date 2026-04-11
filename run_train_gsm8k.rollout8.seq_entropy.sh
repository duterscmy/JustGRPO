#!/bin/bash
#SBATCH --job-name="train_gsm8k_8"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # 请求8块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

output_dir=./checkpoints_gsm8k_num_generation8_test_block1_seq_entropy_only_rollout
mkdir -p $output_dir

#--resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028

accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.rollout8.seq_entropy.py \
  --run_dir $output_dir \
  --block_size 1 \
  --grad_accum 1 #>> $output_dir.log 2>&1