
accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train_aime2024.py \
  --run_dir ./checkpoints_aime2024_num_generation8 \
  --grad_accum 1