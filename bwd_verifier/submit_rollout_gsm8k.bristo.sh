

for block in 1 32 256; do
  for t in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    sbatch rollout_gsm8k.bristo.sh $block $t
  done
done