#!/bin/bash
#SBATCH --job-name="ttrl"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4          # 增加到4，提高数据加载效率
#SBATCH --mem=128G                  # 改用总内存更可靠
#SBATCH --gres=gpu:1                # 请求8块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err



### 激活conda环境
source ~/miniconda3/bin/activate ttrl  # 你的环境名

### 运行debug脚本
bash debug.sh