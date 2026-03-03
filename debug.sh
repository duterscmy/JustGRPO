
accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train.py \
  --run_dir ./checkpoints \
  --grad_accum 16
#   --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028