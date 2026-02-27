#!/bin/bash
### Job Name ###
#SBATCH --job-name="ttrl"

### CPU/Node requirements ###
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8      # 一般跑深度学习一个任务就够了
#SBATCH --cpus-per-task=1       # 明确指定CPU核心数

### CPU Memory (RAM) requirements ###
#SBATCH --mem-per-cpu 8g             # 根据你的任务调整，debug模式20G应该够

### GPU requirements ###
#SBATCH --gpus=8                # 使用1块GPU
#SBATCH --time=24:00:00

### Job log files ###
#SBATCH -o slurm.%j.%N.out        # 输出日志 (%j=jobID, %N=node)
#SBATCH -e slurm.%j.%N.err        # 错误日志



### 激活conda环境
source ~/miniconda3/bin/activate ttrl  # 你的环境名

### 运行debug脚本
bash debug.sh