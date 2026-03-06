#!/bin/bash
#SBATCH --job-name="eval_aime"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate llada

model_path=$1

cp /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct/config.json /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct/*py /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct/*token* $model_path
mkdir -p eval_results

# 1. 规范化路径（去除末尾斜杠）
clean_path=$(echo $model_path | sed 's:/*$::')

# 2. 提取目录名（倒数第二层）和文件名（最底层）
# 假设 model_path 为 /path/to/checkpoints_gsm8k_num_generation8/ckpt-000007
parent_dir=$(basename $(dirname "$clean_path"))
base_name=$(basename "$clean_path")

# 3. 创建目标子目录
target_dir="eval_results/${parent_dir}"
mkdir -p "$target_dir"

# 4. 拼接最终的日志路径
log_path="${target_dir}/${base_name}.log"

echo "Logging to: $log_path"

# Use model_args to adjust the sampling arguments for evaluation.
accelerate launch --num_processes 1 \
    /lus/lfs1aip2/projects/public/u6er/mingyu/dllm/dllm/pipelines/llada/eval.py \
    --tasks "aime24" \
    --model "llada" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=$model_path,max_new_tokens=1024,steps=1024,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"


echo "Evaluation completed!"