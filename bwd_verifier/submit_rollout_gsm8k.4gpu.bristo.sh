#!/bin/bash
#SBATCH --job-name="rollout_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
seed=113

# 所有任务的参数组合 (block, temp)
TASKS=(
  "1   0.4"
  "1   0.6"
  "1   0.8"
  "1   1.0"
)

# TASKS=(
#   "1   0.4"
#   "1   0.6"
#   "1   0.8"
#   "1   1.0"
#   "32  0.4"
#   "32  0.6"
#   "32  0.8"
#   "32  1.0"
#   "32  1.2"
# )

# 单个任务函数：绑定到指定 GPU 运行
run_one() {
  local gpu=$1
  local block=$2
  local t=$3
  local outfile="gsm8k_results.1019.add_records.${block}.${t}.seed${seed}.json"
  local logfile="logs/gpu${gpu}.1019.block${block}.temp${t}.log"

  echo "[GPU ${gpu}] Starting: block=${block} temp=${t}"
  CUDA_VISIBLE_DEVICES=${gpu} python rollout_gsm8k.py \
    --steps 256 \
    --gen_length 256 \
    --block_length ${block} \
    --temperature ${t} \
    --remasking low_confidence \
    --num_rollouts 8 \
    --max_problems -1019 \
    --output_file ${outfile} \
    --verbose \
    --model_path /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct \
    --device cuda \
    --seed ${seed} \
    > ${logfile} 2>&1
  echo "[GPU ${gpu}] Done: block=${block} temp=${t}"
}

# 4 个 GPU 并行，每批跑 4 个任务，等待全部完成再开下一批
total=${#TASKS[@]}
i=0

while [ $i -lt $total ]; do
  pids=()
  for gpu in 0 1 2 3; do
    if [ $i -lt $total ]; then
      read block t <<< "${TASKS[$i]}"
      run_one $gpu $block $t &
      pids+=($!)
      i=$((i + 1))
    fi
  done
  # 等待这一批 4 个任务全部完成
  for pid in "${pids[@]}"; do
    wait $pid
  done
  echo "=== Batch done, completed $i / $total tasks ==="
done

echo "All $total tasks completed!"