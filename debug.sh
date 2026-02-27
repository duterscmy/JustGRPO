export FSDP_WRITEBACK_PARAMS=0

accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train.py \
  --run_dir ./checkpoints \
  --grad_accum 16