python -m llama_recipes.inference.checkpoint_converter_fsdp_hf \
    --fsdp_checkpoint_path /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/training-state-000008 \        # 你的checkpoints目录
    --consolidated_model_path /lus/lfs1aip2/projects/public/u6er/mingyu/justGRPO/checkpoints/08_hf \     # 输出目录
    --HF_model_path_or_name /lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct  # 原始模型配置