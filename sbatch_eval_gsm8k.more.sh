#!/bin/bash
#SBATCH --job-name="eval_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                # 请求2块GPU
#SBATCH --time=5:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

model_path=$1
length=$2
block=$3
mkdir -p eval_results

# 检查model_path是否包含LLaDA关键字
if [[ "$model_path" != *"LLaDA"* ]]; then
    echo "model_path does not contain 'LLaDA', copying config files..."

    # 源路径
    source_model_path="/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct"

    # 检查源路径是否存在
    if [ -d "$source_model_path" ]; then
        # 复制所有.py, .json, .txt文件到model_path
        cp "$source_model_path"/*.py "$source_model_path"/*token*  "$model_path"/ 2>/dev/null || true
        echo "Copied .py, .json, .txt files from $source_model_path to $model_path"
    else
        echo "Warning: Source path $source_model_path does not exist!"
    fi
else
    echo "model_path contains 'LLaDA', no need to copy config files."
fi


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
log_path="${target_dir}/${base_name}.gsm8k.${length}.${block}.log"

echo "Logging to: $log_path"

# 5. 运行评估
torchrun --standalone --nproc-per-node=2 eval.py \
  --ckpt_path "$model_path" \
  --steps $length \
  --gen_length $length \
  --block_length $block &> "$log_path"


echo "Evaluation completed!"