#!/bin/bash
#SBATCH --job-name="train_math"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

t=0.6
block=1
lr=5e-6
max_level=3
output_dir=./checkpoints_math500_num_generation8_block${block}_t${t}_lr${lr}_level1_${max_level}_0419_batch64

#   --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_math500_num_generation8_block1_t0.6_lr1e-6/training-state-000005 \

mkdir -p $output_dir
accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_math500.py \
  --run_dir $output_dir \
  --temperature ${t} \
  --lr $lr \
  --block_size $block \
  --max_level $max_level \
  --total_steps 10 --save_every 5 \
  --grad_accum 16 >> $output_dir.log 2>&1