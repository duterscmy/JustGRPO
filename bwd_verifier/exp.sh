input=$1
output=$2

# --strategies first majority \
#         highest_confidence weighted_confidence \
#         vc_alpha vb_alpha vcb_alpha voting_then_vcb \

python exp.py "$input" "$output" \
    --strategies vb_alpha vcb_alpha voting_then_vcb \
    --alpha 0.05 --beta 0.03 --gamma 0.99 \
    --confidence_scale_low 0.9 -n 16 --backward_key "probability"