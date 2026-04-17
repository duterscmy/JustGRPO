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

output_dir=./checkpoints_gsm8k_num_generation8_test_block1_0315
mkdir -p $output_dir

#--resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.rollout8.py \
  --run_dir $output_dir \
  --block_size 1 \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_gsm8k_num_generation8_test_block1_0315/training-state-000020 \
  --total_steps 40 --save_every 5 \
  --grad_accum 8 >> $output_dir.log 2>&1