#!/bin/bash
#SBATCH --job-name="eval_math500"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

model_path=$1
length=$2
block=$3
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
log_path="${target_dir}/${base_name}.math500.${length}.${block}.log"

echo "Logging to: $log_path"


# Use model_args to adjust the sampling arguments for evaluation.
# accelerate launch --num_processes 1 \
#     /lus/lfs1aip2/projects/public/u6er/mingyu/dllm/dllm/pipelines/llada/eval.py \
#     --tasks "minerva_math500" \
#     --model "llada" \
#     --apply_chat_template \
#     --num_fewshot 0 \
#     --output_path ${target_dir}/${base_name} \
#     --model_args "pretrained=$model_path,max_new_tokens=256,steps=256,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"


torchrun --standalone --nproc-per-node=2 eval.py \
  --ckpt_path "$model_path" \
  --task math500 \
  --steps $length \
  --gen_length $length \
  --block_length $block &> "$log_path"


echo "Evaluation completed!"