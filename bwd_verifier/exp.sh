input=$1
output=$2
python exp.py "$input" "$output" \
    --strategies first majority \
        highest_confidence weighted_confidence confidence_filter \
        vc_arithmetic vc_geometric vc_alpha \
        vb_arithmetic vb_geometric vb_alpha \
        vcb_arithmetic vcb_geometric vcb_alpha \
        fobar \
        voting_then_conf voting_then_backward voting_then_vcb \
    --alpha 0.5 --beta 0.3 --gamma 0.2 \
    --confidence_scale_low 0.9 -n 16