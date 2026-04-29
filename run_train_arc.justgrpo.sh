#!/bin/bash
#SBATCH --job-name="train_arc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
# source ~/.bashrc # 你的环境名
# conda activate ttrl

block=32
t=1.0
lr=1e-6
length=128
output_dir=./checkpoints_arc_num_generation8_test_length${length}_block${block}_t${t}_lr${lr}_0428_batch64_justgrpo
mkdir -p $output_dir

#--resume_ckpt 
accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train_arc.justgrpo.py \
  --model_path /gpfs/home5/xiaoq/caomingyu/models/LLaDA-8B-Instruct \
  --run_dir $output_dir \
  --block_size $block \
  --grad_accum 1 \
  --lr $lr \
  --total_steps 30 --save_every 5 \
  --temperature $t \
  --gen_length $length \
  --gen_steps $length #>> $output_dir.log 2>&1