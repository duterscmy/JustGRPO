input=$1
base=$(basename "$input" .json)
python draw_metric_acc.py "$input" \
    --metrics confidence backward_digits backward_probability \
    --n_buckets 10 \
    --save_dir "${base}_fig" \
    --dataset_name "$base"