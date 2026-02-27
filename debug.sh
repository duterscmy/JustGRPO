accelerate launch --num_processes 8 --main_process_ip localhost --config_file configs/fsdp.yaml train.py \
  --run_dir ./checkpoints \
  --grad_accum 8