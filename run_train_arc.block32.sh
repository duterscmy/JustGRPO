#!/bin/bash
#SBATCH --job-name="train_arc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

block=32
t=1.0
lr=1e-6
length=64
output_dir=./checkpoints_arc_num_generation8_test_length${length}_block${block}_t${t}_lr${lr}_0425_batch64
mkdir -p $output_dir

#--resume_ckpt 
accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_arc.py \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_arc_num_generation8_test_block32_t1.0_lr1e-6_0418_batch64/training-state-000030 \
  --run_dir $output_dir \
  --block_size $block \
  --grad_accum 16 \
  --lr $lr \
  --total_steps 45 --save_every 5 \
  --temperature $t \
  --gen_length $length \
  --gen_steps $length >> $output_dir.log 2>&1