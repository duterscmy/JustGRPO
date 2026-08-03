#!/bin/bash
set -euo pipefail

target_dir=${1:-dataset}
mkdir -p "$target_dir"

# Pin the d1 repository revision used when this integration was prepared.
revision="6f5abf5ca8a58c6e08bbf06d412ad260dca6dbd3"
base_url="https://raw.githubusercontent.com/dllm-reasoning/d1/${revision}/dataset"

for filename in 4x4_test_sudoku.csv countdown_cd3_test.jsonl; do
  tmp_path="${target_dir}/${filename}.tmp"
  curl --fail --location --retry 3 \
    "${base_url}/${filename}" \
    --output "$tmp_path"
  test -s "$tmp_path"
  mv "$tmp_path" "${target_dir}/${filename}"
  echo "Downloaded ${target_dir}/${filename}"
done
