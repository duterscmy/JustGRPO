#!/bin/bash
# evaluate_deepseek_math500_zeroshot.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 根据你的GPU数量调整
export TOKENIZERS_PARALLELISM=false

# Zero-shot评估命令
lm_eval --model hf \
    --model_args pretrained=deepseek-ai/deepseek-math-7b-instruct,trust_remote_code=True,dtype=bfloat16 \
    --tasks minerva_math500 \
    --batch_size 4 \
    --output_path ./eval_results/deepseek_math500_zeroshot \
    --log_samples \
    --apply_chat_template \
    --num_fewshot 0  # 显式设置为0，表示zero-shot