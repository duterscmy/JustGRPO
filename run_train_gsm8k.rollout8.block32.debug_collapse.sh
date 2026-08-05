#!/bin/bash
#SBATCH --job-name="train_gsm8k_8"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc
conda activate ttrl

block=32
t=1.0
lr=5e-6
kl=0.01

output_dir=./checkpoints_gsm8k_num_generation8_test_block${block}_temperature${t}_lr${lr}_kl${kl}_noscale_0521_debug_collapse
mkdir -p $output_dir

# --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.rollout8.debug_collapse.py \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints_gsm8k_num_generation8_test_block32_temperature1.0_lr5e-6_kl0.01_noscale_0521_debug_collapse/training-state-000040 \
  --run_dir $output_dir \
  --block_size $block \
  --lr $lr \
  --temperature $t \
  --kl_beta $kl \
  --no_scale_by_grad_accum \
  --total_steps 60 \
  --save_every 5 \
  --policy_shift_stride 8 \
  --grad_accum 8 >> $output_dir.log 2>&1