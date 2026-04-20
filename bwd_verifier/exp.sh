# 跑所有组合策略
input=$1
output=$2
python evaluate_strategies.py "$input" "$output" \
    --strategies majority weighted_confidence vcb_geometric vcb_alpha voting_then_vcb \
    --alpha 0.5 --beta 0.3 --gamma 0.2 \
    --confidence_scale_low 0.9