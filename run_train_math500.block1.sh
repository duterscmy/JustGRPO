#!/bin/bash
#SBATCH --job-name="train_math"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc
conda activate ttrl

t=0.6
block=1
lr=5e-6
max_level=5
grad_accum=8
gain=1.0
max_grad_norm=1.0

output_dir=./checkpoints_math500_num_generation8_block${block}_t${t}_lr${lr}_level1_${max_level}_accum${grad_accum}_gain${gain}_clip${max_grad_norm}_debug
mkdir -p $output_dir

# Resume example:
# --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_math500_num_generation8_block1_t0.6_lr1e-6/training-state-000005

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_math500.py \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_math500_num_generation8_block1_t0.6_lr5e-6_level1_5_accum8_gain1.0_clip1.0_debug/training-state-000020 \
  --run_dir $output_dir \
  --temperature ${t} \
  --lr $lr \
  --block_size $block \
  --max_level $max_level \
  --total_steps 40 \
  --save_every 5 \
  --grad_accum $grad_accum \
  --gain $gain \
  --max_grad_norm $max_grad_norm \
  >> $output_dir.log 2>&1