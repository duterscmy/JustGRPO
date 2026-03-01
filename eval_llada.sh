#!/bin/bash
#SBATCH --job-name="ttrl"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                # 请求8块GPU
#SBATCH --time=3:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err



### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

model_path=$1
mkdir -p eval_results
torchrun --standalone --nproc-per-node=2 eval.py \
  --ckpt_path $model_path \
  --steps 256 \
  --gen_length 256 \
  --block_length 32 &> eval_results/$(basename $model_path).log