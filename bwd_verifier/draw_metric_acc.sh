input=$1
base=$(basename "$input" .json)
python plot_score_vs_accuracy.py "$input" \
    --metrics confidence backward_digits backward_probability \
    --n_buckets 10 \
    --save_dir "${base}_fig" \
    --dataset_name "$base"