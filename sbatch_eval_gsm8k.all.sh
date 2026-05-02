#!/bin/bash
#SBATCH --job-name="eval_gsm8k_multi"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

source ~/.bashrc
conda activate ttrl

mkdir -p eval_results

source_model_path="/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct"

model_paths=(
  "checkpoints_gsm8k_num_generation8_test_block32_temperature1.4_lr5e-6_0501/ckpt-000005"
  "checkpoints_gsm8k_num_generation8_test_block32_temperature1.4_lr5e-6_0501/ckpt-000010"
  "checkpoints_gsm8k_num_generation8_test_block32_temperature1.4_lr5e-6_0501/ckpt-000015"
  "checkpoints_gsm8k_num_generation8_test_block32_temperature1.4_lr5e-6_0501/ckpt-000020"
)

for model_path in "${model_paths[@]}"; do
    echo "============================================================"
    echo "Evaluating model: $model_path"
    echo "============================================================"

    if [ ! -d "$model_path" ]; then
        echo "Warning: model path does not exist: $model_path"
        echo "Skipping..."
        continue
    fi

    # If checkpoint directory does not contain LLaDA files, copy tokenizer/config/code files.
    if [[ "$model_path" != *"LLaDA"* ]]; then
        echo "model_path does not contain 'LLaDA', copying config/tokenizer files..."

        if [ -d "$source_model_path" ]; then
            cp "$source_model_path"/*.py "$source_model_path"/*token* "$model_path"/ 2>/dev/null || true
            echo "Copied files from $source_model_path to $model_path"
        else
            echo "Warning: source path $source_model_path does not exist!"
        fi
    else
        echo "model_path contains 'LLaDA', no need to copy config files."
    fi

    clean_path=$(echo "$model_path" | sed 's:/*$::')
    parent_dir=$(basename "$(dirname "$clean_path")")
    base_name=$(basename "$clean_path")

    target_dir="eval_results/${parent_dir}"
    mkdir -p "$target_dir"

    log_path="${target_dir}/${base_name}.gsm8k.log"

    echo "Logging to: $log_path"

    torchrun --standalone --nproc-per-node=4 eval.py \
      --ckpt_path "$model_path" \
      --steps 256 \
      --gen_length 256 \
      --block_length 32 &> "$log_path"

    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "Evaluation failed for $model_path with exit code $exit_code"
    else
        echo "Evaluation completed for $model_path"
    fi

    echo ""
done

echo "All evaluations completed!"