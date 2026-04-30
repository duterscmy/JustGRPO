
for block in 1; do
  for tmp in 1.2; do
    sbatch rollout_gsm8k.qiao.sh $block $tmp
  done
done