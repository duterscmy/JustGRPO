
output_dir=./checkpoints_math500_num_generation8_trainset
mkdir -p $output_dir
accelerate launch --num_processes 1 --main_process_ip localhost --config_file configs/fsdp.yaml train_math500_trainset.py \
  --run_dir $output_dir \
  --grad_accum 1 #>> $output_dir.log 2>&1