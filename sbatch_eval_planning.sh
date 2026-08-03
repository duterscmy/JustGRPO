#!/bin/bash
#SBATCH --job-name="eval_planning"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -euo pipefail

source ~/.bashrc
conda activate soar

model_path=${1:?"Usage: sbatch $0 <model_path> <sudoku|countdown> [length] [block] [data_dir]"}
task=${2:?"Usage: sbatch $0 <model_path> <sudoku|countdown> [length] [block] [data_dir]"}
length=${3:-256}
block=${4:-32}
data_dir=${5:-${DATA_DIR:-dataset}}
base_model_path=${BASE_MODEL_PATH:-/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct}

if [[ "$task" != "sudoku" && "$task" != "countdown" ]]; then
  echo "task must be sudoku or countdown" >&2
  exit 2
fi

# Full-weight checkpoints may omit custom modeling/tokenizer source files.
# Copy only missing assets, preserving checkpoint-specific files.
shopt -s nullglob
for src in \
  "$base_model_path"/*.py \
  "$base_model_path"/*token* \
  "$base_model_path"/config.json \
  "$base_model_path"/generation_config.json; do
  dest="$model_path/$(basename "$src")"
  if [[ ! -e "$dest" ]]; then
    cp "$src" "$dest"
  fi
done

clean_path=${model_path%/}
parent_dir=$(basename "$(dirname "$clean_path")")
base_name=$(basename "$clean_path")
target_dir="eval_results/${parent_dir}"
mkdir -p "$target_dir"
result_path="${target_dir}/${base_name}.${task}.${length}.${block}.json"
log_path="${target_dir}/${base_name}.${task}.${length}.${block}.log"

python eval_planning.py \
  --task "$task" \
  --model_path "$model_path" \
  --data_dir "$data_dir" \
  --output_path "$result_path" \
  --gen_length "$length" \
  --steps "$length" \
  --block_size "$block" \
  --temperature 0.0 \
  --max_samples 256 \
  > "$log_path" 2>&1

echo "Evaluation completed: $result_path"
