#!/bin/bash
#SBATCH --job-name="train_arc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=16G
#SBATCH --partition=gpu_h100

# Create logs directory if it doesn't exist
mkdir -p logs
source /gpfs/home5/xiaoq/.bashrc
conda activate ttrl
cd /home/xiaoq/caomingyu/justgrpo

block=32
t=1.0
lr=1e-6
length=128
output_dir=./checkpoints_arc_num_generation8_test_length${length}_block${block}_t${t}_lr${lr}_0429_batch64_justgrpo
mkdir -p $output_dir

#--resume_ckpt 
accelerate launch --num_processes 2 --main_process_ip localhost --config_file configs/fsdp.yaml train_arc.justgrpo.py \
  --model_path /gpfs/home5/xiaoq/caomingyu/models/LLaDA-8B-Instruct \
  --run_dir $output_dir \
  --block_size $block \
  --grad_accum 32 \
  --lr $lr \
  --total_steps 30 --save_every 5 \
  --temperature $t \
  --gen_length $length \
  --gen_steps $length >> $output_dir.log 2>&1