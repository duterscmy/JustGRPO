
output_dir=./checkpoints
mkdir -p $output_dir

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.py \
  --run_dir ./checkpoints \
  --grad_accum 16 \
  --resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028 >> $output_dir.log 2>&1