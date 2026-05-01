#!/bin/bash
#SBATCH --job-name="train_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu_h100

# Create logs directory if it doesn't exist
mkdir -p logs
source /gpfs/home5/xiaoq/.bashrc
conda activate ttrl
cd /home/xiaoq/caomingyu/justgrpo

block=1
t=0.5
lr=5e-6
output_dir=./checkpoints_gsm8k_num_generation8_test_block${block}_temperature${t}_lr${lr}_0501
mkdir -p $output_dir

#--resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.rollout8.py \
  --model_path /gpfs/home5/xiaoq/caomingyu/models/LLaDA-8B-Instruct \
  --run_dir $output_dir \
  --block_size $block \
  --lr $lr  \
  --temperature $t \
  --total_steps 45 --save_every 5 \
  --grad_accum 8 >> $output_dir.log 2>&1