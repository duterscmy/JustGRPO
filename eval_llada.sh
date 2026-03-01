#!/bin/bash
#SBATCH --job-name="ttrl"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                # 请求2块GPU
#SBATCH --time=3:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl

model_path=$1
mkdir -p eval_results

# 检查model_path是否包含LLaDA关键字
if [[ "$model_path" != *"LLaDA"* ]]; then
    echo "model_path does not contain 'LLaDA', copying config files..."
    
    # 源路径
    source_model_path="/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct"
    
    # 检查源路径是否存在
    if [ -d "$source_model_path" ]; then
        # 复制所有.py, .json, .txt文件到model_path
        cp "$source_model_path"/*.py "$model_path"/ 2>/dev/null || true
        echo "Copied .py, .json, .txt files from $source_model_path to $model_path"
    else
        echo "Warning: Source path $source_model_path does not exist!"
    fi
else
    echo "model_path contains 'LLaDA', no need to copy config files."
fi

# 运行评估
torchrun --standalone --nproc-per-node=2 eval.py \
  --ckpt_path $model_path \
  --steps 256 \
  --gen_length 256 \
  --block_length 32 &> eval_results/$(basename $model_path).log

echo "Evaluation completed!"