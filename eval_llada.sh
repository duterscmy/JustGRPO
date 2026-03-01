
model_path=$1
mkdir -p eval_results
torchrun --standalone --nproc-per-node=3 eval.py \
  --ckpt_path $model_path \
  --steps 256 \
  --gen_length 256 \
  --block_length 32
# &> eval_results/$(basename $model_path).log