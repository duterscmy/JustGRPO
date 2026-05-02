#!/bin/bash
#SBATCH --job-name="train_math500_dttrl"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
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

t=0.6
block=32
lr=5e-6
max_level=5
output_dir=/scratch-shared/xiaoq/caomingyu/models/checkpoints_math500_num_generation8_block${block}_t${t}_lr${lr}_level1_${max_level}_0429_batch64_weighted_confidence

#   --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_math500_num_generation8_block1_t0.6_lr1e-6/training-state-000005 \

mkdir -p $output_dir
accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_math500.rank.py \
  --model_path /gpfs/home5/xiaoq/caomingyu/models/LLaDA-8B-Instruct \
  --run_dir $output_dir \
  --temperature ${t} \
  --lr $lr \
  --block_size $block \
  --max_level $max_level \
  --total_steps 15 --save_every 5 \
  --grad_accum 16 >> $output_dir.log 2>&1