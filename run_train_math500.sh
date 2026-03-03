
accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train_math500.py \
  --run_dir ./checkpoints_math500_num_generation8 \
  --grad_accum 2