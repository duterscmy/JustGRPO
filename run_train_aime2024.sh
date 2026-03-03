output_dir=./checkpoints_aime2024_num_generation8
mkdir -p $output_dir
accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_aime2024.py \
  --run_dir ./checkpoints_aime2024_num_generation8 \
  --grad_accum 16 >> $output_dir.log 2>&1