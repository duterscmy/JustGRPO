
output_dir=./checkpoints_gsm8k_num_generation16
mkdir -p $output_dir

#--resume_ckpt /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000028

accelerate launch --num_processes 4 --main_process_ip localhost --config_file configs/fsdp.yaml train_gsm8k.py \
  --run_dir $output_dir \
  --grad_accum 16 >> $output_dir.log 2>&1