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

block=32
t=1.0
lr=5e-6
output_dir=./checkpoints_gsm8k_num_generation8_test_block${block}_temperature${t}_lr${lr}_0424_rank
mkdir -p $output_dir

#--resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.rollout8.rank.py \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_gsm8k_num_generation8_test_block32_temperature1.0_lr5e-6_0424_rank/training-state-000010 \
  --run_dir $output_dir \
  --block_size 32 \
  --lr $lr  \
  --temperature 1.0 \
  --total_steps 30 --save_every 2 \
  --grad_accum 8 >> $output_dir.log 2>&1