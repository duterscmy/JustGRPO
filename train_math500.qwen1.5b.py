import os
import re
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List

import utils.distributed as dist
from grpo import sample, sample_with_repeat, logprob_loss, compute_group_advantages

def generate_with_qwen(model, tokenizer, prompts, device, num_generations=1, temperature=0.6, max_new_tokens=256):
    """
    使用Qwen模型的generate方法生成回复
    
    Args:
        model: Qwen模型
        tokenizer: 对应的tokenizer
        prompts: 提示词列表
        device: 设备
        num_generations: 每个prompt生成的回复数
        temperature: 采样温度
        max_new_tokens: 最大新token数
    """
    # 应用聊天模板
    formatted_prompts = []
    for p in prompts:
        formatted_prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False
        ))
    
    # 编码所有prompts
    inputs = tokenizer(formatted_prompts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # 重复输入以生成多个样本
    input_ids = inputs['input_ids'].repeat_interleave(num_generations, dim=0)
    attention_mask = inputs['attention_mask'].repeat_interleave(num_generations, dim=0)
    
    # 生成配置
    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': temperature > 0,
        'temperature': temperature if temperature > 0 else 1.0,
        'top_p': 0.95 if temperature > 0 else None,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
    }
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    
    return generated_ids

def sample_with_repeat_qwen(model, batch, tokenizer, device, reward_fn=None, 
                           num_generations=1, temperature=0.6, max_new_tokens=256, 
                           repeat_time=1):
    """
    使用Qwen的generate方法采样的版本
    """
    prompts = batch['problems']  # 原始问题列表
    
    generate_ids_list = []
    
    for _ in range(repeat_time):
        generated_ids = generate_with_qwen(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            num_generations=num_generations,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        generate_ids_list.append(generated_ids)
        
        # 清理缓存
        torch.cuda.empty_cache()
    
    all_generated_ids = torch.cat(generate_ids_list, dim=0)
    
    # 解码生成的回复
    responses = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)
    
    # 计算每个回复的奖励
    rewards = reward_fn(batch, responses, num_generations * repeat_time, device).float()
    
    return {
        'generated_ids': all_generated_ids,
        'prompt_len': tokenizer(prompts[0], return_tensors='pt')['input_ids'].shape[1],  # 近似prompt长度
        'rewards': rewards,
    }


@dataclass
class TrainConfig:
    """Training hyperparameters for GRPO."""
    
    # --- Model ---
    model_path: str = "Qwen/Qwen2.5-Math-1.5B"
    
    # --- Training ---
    batch_size_per_device: int = 1
    grad_accumulation: int = 8
    total_steps: int = 5
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 1234
    num_generations: int = 4
    repeat_times: int = 1
    sample_repeat_times: int = 2
    max_new_tokens: int = 256  # 替换原来的gen_steps和gen_length
    temperature: float = 0.6
    # 移除block_size参数

    # --- Misc ---
    output_dir: str = "./checkpoints_math500_qwen_math_1.5b_num_generation{}".format(num_generations)
    log_every: int = 1
    save_every: int = 5
    resume_ckpt: Optional[str] = None


def train(config: TrainConfig):
    """
    Main GRPO training loop.
    
    Args:
        config: TrainConfig with model path, learning rate, batch size, etc.
    """
    
    # --- Initialize distributed ---
    dist.init()
    rank = dist.get_rank()
    device = torch.device('cuda')
    
    print("=" * 60)
    print("JustGRPO Training with Qwen generate")
    print("=" * 60)

    # --- Random seeds ---
    np.random.seed((config.seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    # --- Load model ---
    print(f"Loading model from {config.model_path}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM  # 改为AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.eval().to(device)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token
    
    # --- Load dataset ---
    print("Loading dataset...")
    from data.math import load_math500_dataset_and_reward
    
    dataloader, reward_fn = load_math500_dataset_and_reward(
        local_path="HuggingFaceH4/MATH-500",
        split='test',
        batch_size=config.batch_size_per_device,
        num_workers=4,
    )
    
    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    
    # --- Accelerator setup ---
    accelerator = dist.get_accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # --- Resume ---
    start_step = 0
    if config.resume_ckpt is not None:
        print(config.resume_ckpt)
        local_resume_path = config.resume_ckpt
        if os.path.exists(local_resume_path):
            print(f"Resuming from {local_resume_path}")
            accelerator.load_state(local_resume_path)
        match = re.search(r'(\d+)$', config.resume_ckpt.rstrip('/'))
        if match:
            start_step = int(match.group(1))
            print("start_step is {}".format(start_step))
    
    dataloader_iter = iter(dataloader)
    
    if start_step > 0:
        print(f"Skipping {start_step} batches...")
        for _ in range(start_step):
            next(dataloader_iter)
    
    # --- Output directory ---
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
    
    # --- Training loop ---
    print(f"Starting training for {config.total_steps} steps...")
    print(f"Group size: {config.num_generations * config.repeat_times * config.sample_repeat_times}")
    print(f"Grad accumulation: {config.grad_accumulation}")
    print(f"Effective batch: {config.batch_size_per_device * dist.get_world_size() * config.grad_accumulation}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max new tokens: {config.max_new_tokens}")
    print(f"Temperature: {config.temperature}")

    for step in range(start_step, config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        
        all_rewards = []
        
        for accum_idx in range(config.grad_accumulation):
            print(f"[Step {step+1}/{config.total_steps}] [Accum {accum_idx+1}/{config.grad_accumulation}] Sampling...")
            
            with dist.ddp_sync(model, sync=(accum_idx == config.grad_accumulation - 1)):
                model.eval()
                
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    # --- Rollout with Qwen generate ---
                    batch = next(dataloader_iter)
                    inputs_chunks = []
                    
                    print(f"Using temperature: {config.temperature}")
                    
                    for _ in range(config.repeat_times):
                        inputs = sample_with_repeat_qwen(
                            model=model,
                            batch=batch,
                            tokenizer=tokenizer,
                            device=device,
                            reward_fn=reward_fn,
                            temperature=config.temperature,
                            num_generations=config.num_generations,
                            max_new_tokens=config.max_new_tokens,
                            repeat_time=config.sample_repeat_times
                        )
                        inputs_chunks.append(inputs)
                        torch.cuda.empty_cache()

                    # --- Compute Advantages ---
                    rewards = torch.cat([chunk['rewards'] for chunk in inputs_chunks], dim=0)
                    advantages = compute_group_advantages(
                        rewards, 
                        config.num_generations * config.repeat_times * config.sample_repeat_times
                    )
                    valid_samples = (advantages != 0).sum()
                    split_advantages = advantages.split(config.num_generations * config.sample_repeat_times, dim=0)
                    
                    for chunk, adv in zip(inputs_chunks, split_advantages):
                        chunk["advantages"] = adv
                    
                    accelerator.wait_for_everyone()

                # 这里可以取消注释损失计算和优化步骤
                # ...
        
        print(f"[Step {step+1}/{config.total_steps}] Completed")
    
    print("\nTraining complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="JustGRPO Training with Qwen generate")
    
    parser.add_argument("--run_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--temperature", type=float, default=0.6, help="rollout temperature")
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create config from CLI args
    config = TrainConfig(
        output_dir=args.run_dir,
        grad_accumulation=args.grad_accum,
        resume_ckpt=args.resume_ckpt,
        temperature=args.temperature,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,  # 新的参数
    )

    train(config)