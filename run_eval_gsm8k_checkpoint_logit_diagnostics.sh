#!/bin/bash
#SBATCH --job-name="gsm8k_logit_diag"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err


if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 TRAINING_DIRECTORY" >&2
    exit 2
fi

source ~/.bashrc
conda activate ttrl

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="${1%/}"

if [[ ! -d "$run_dir" ]]; then
    echo "Training directory not found: $run_dir" >&2
    exit 2
fi

base_model_path="${BASE_MODEL_PATH:-/lus/lfs1aip2/projects/public/u6os/mingyu/models/LLaDA-8B-Instruct}"
output_dir="${OUTPUT_DIR:-${run_dir}_logit_diagnostics}"

mkdir -p "$output_dir"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

python eval_gsm8k_checkpoint_logit_diagnostics.py \
    --run_dir "$run_dir" \
    --base_model_path "$base_model_path" \
    --output_dir "$output_dir" \
    --num_questions "${NUM_QUESTIONS:-128}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --mask_ratio "${MASK_RATIO:-0.5}" \
    --js_positions_per_question "${JS_POSITIONS_PER_QUESTION:-8}" \
    --js_chunk_size "${JS_CHUNK_SIZE:-4}" \
    --max_answer_tokens "${MAX_ANSWER_TOKENS:-256}" \
    --max_sequence_length "${MAX_SEQUENCE_LENGTH:-512}" \
    --probe_seed "${PROBE_SEED:-2026}" \
    --mask_seed "${MASK_SEED:-314159}" \
    --dtype "${DTYPE:-bfloat16}" \
    >"$output_dir/evaluation.log" 2>&1

echo "Finished. Results: $output_dir"
