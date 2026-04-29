
for block in 1 32 256; do
  for tmp in 0.4 0.6 0.8 1.0 1.2; do
    sbatch rollout_gsm8k.qiao.sh $block $tmp
  done
done