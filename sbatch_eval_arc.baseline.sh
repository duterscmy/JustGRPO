#!/bin/bash
#SBATCH --job-name="eval_arc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # 请求2块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
#  --limit 256
length=128
block=16
temperature=0.0
accelerate launch --num_processes 1 eval_llada.py \
  --tasks arc_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path eval_results/baseline/arc-${length}-${block}-${temperature} --log_samples \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',temperature=${temperature},enable_early_exit=false,enable_soar=false,gen_length=${length},steps=${length},block_length=${block},answer_length=5,torch_dtype=torch.bfloat16 &> logs/baseline-arc-ns0-length${length}-block${block}-temperature${temperature}.log