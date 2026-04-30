input=$1
base=$(basename "$input" .json)
python draw_metric_acc.py "$input" \
    --metric confidence\
    --n_buckets 6 \
    --save_path "${base}_fig" \
    --dataset_names GSM8K