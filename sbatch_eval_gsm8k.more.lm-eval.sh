#!/bin/bash
#SBATCH --job-name="eval_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

### 激活conda环境
source ~/.bashrc
conda activate soar

model_path=$1
cp /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct/config.json /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct/*py /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct/*token* $model_path
mkdir -p eval_results

# 1. 规范化路径（去除末尾斜杠）
clean_path=$(echo $model_path | sed 's:/*$::')

# 2. 提取目录名（倒数第二层）和文件名（最底层）
parent_dir=$(basename $(dirname "$clean_path"))
base_name=$(basename "$clean_path")

# 3. 创建目标子目录
target_dir="eval_results/${parent_dir}"
mkdir -p "$target_dir"

length=$2
block=$3
temperature=0.0
# 4. 拼接最终的日志路径
result_path="${target_dir}/${base_name}.gsm8k.lm-eval.${length}.${block}"
log_path="${target_dir}/${base_name}.gsm8k.lm-eval.${length}.${block}.log"

echo "Logging to: $log_path"

# 设置参数



# 运行评估
accelerate launch --num_processes 1 eval_llada.py \
  --tasks gsm8k_cot_zeroshot \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path $result_path --log_samples \
  --model_args model_path=${model_path},temperature=${temperature},enable_early_exit=false,enable_soar=false,gen_length=${length},steps=${length},block_length=${block},answer_length=5,torch_dtype=torch.bfloat16 &> $log_path

echo "Evaluation completed!"