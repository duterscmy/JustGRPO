#!/bin/bash
#SBATCH --job-name="ttrl"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # 请求2块GPU
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
        cp "$source_model_path"/*.py "$source_model_path"/*token*  "$model_path"/ 2>/dev/null || true
        echo "Copied .py, .json, .txt files from $source_model_path to $model_path"
    else
        echo "Warning: Source path $source_model_path does not exist!"
    fi
else
    echo "model_path contains 'LLaDA', no need to copy config files."
fi


# 先规范化路径（去除末尾的斜杠）
clean_path=$(echo $model_path | sed 's:/*$::')

# 提取最后两层非空目录/文件并拼接
log_name=$(echo $clean_path | awk -F'/' '{
    nf = NF
    if (nf >= 2) {
        print $(nf-1)"_"$nf
    } else if (nf == 1) {
        print $1
    } else {
        print "unknown"
    }
}')

# 运行评估
torchrun --standalone --nproc-per-node=4 eval.py \
  --ckpt_dir $model_path \
  --steps 256 \
  --gen_length 256 \
  --block_length 32 &> eval_results/${log_name}.log

echo "Evaluation completed!"