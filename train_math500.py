import os
import re
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

import utils.distributed as dist
from grpo import sample, sample_with_repeat, logprob_loss, compute_group_advantages


@dataclass
class TrainConfig:
    """Training hyperparameters for GRPO."""

    # --- Model ---
    model_path: str = "/lus/lfs1aip2/projects/public/u6os/mingyu/models/LLaDA-8B-Instruct"

    # --- Training ---
    batch_size_per_device: int = 1
    grad_accumulation: int = 8
    total_steps: int = 10
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 1234

    num_generations: int = 4
    repeat_times: int = 1
    sample_repeat_times: int = 2

    gen_steps: int = 256
    gen_length: int = 256
    temperature: float = 0.6
    block_size: int = 1
    max_level: int = 3

    # --- Loss scaling ---
    gain: float = 1.0
    scale_by_grad_accum: bool = True

    # --- Misc ---
    output_dir: str = "./checkpoints_math500_num_generation4"
    log_every: int = 1
    save_every: int = 5
    resume_ckpt: Optional[str] = None

    only_rollout: int = 0


def get_global_grad_norm_after_clip(model, accelerator=None, norm_type=2.0):
    """
    Compute global gradient norm after clipping.

    In FSDP/DDP, each rank may only see local/sharded gradients.
    We compute local squared norm and gather across ranks.
    """
    params = [p for p in model.parameters() if p.grad is not None]

    if len(params) == 0:
        device = accelerator.device if accelerator is not None else torch.device("cuda")
        return torch.tensor(0.0, device=device)

    device = params[0].grad.device

    if norm_type != 2.0:
        raise NotImplementedError("Only L2 norm is implemented here.")

    local_sq_norm = torch.zeros((), device=device, dtype=torch.float32)

    for p in params:
        grad = p.grad.detach().float()
        local_sq_norm += grad.pow(2).sum()

    if accelerator is not None:
        gathered_sq_norm = accelerator.gather(local_sq_norm.reshape(1))
        global_sq_norm = gathered_sq_norm.sum()
    else:
        global_sq_norm = local_sq_norm

    return global_sq_norm.sqrt()


def train(config: TrainConfig):
    """
    Main GRPO training loop.

    Args:
        config: TrainConfig with model path, learning rate, batch size, etc.
    """

    # --- Initialize distributed ---
    dist.init()
    rank = dist.get_rank()
    device = torch.device("cuda")

    if rank == 0:
        print("=" * 60)
        print("JustGRPO Training")
        print("=" * 60)

    # --- Random seeds ---
    np.random.seed((config.seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # --- Load model ---
    if rank == 0:
        print(f"Loading model from {config.model_path}...")

    from transformers import AutoTokenizer, AutoModel

    model = AutoModel.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.eval().to(device)

    # Activation checkpointing
    if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
        model.model.set_activation_checkpointing("whole_layer")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token_id = 126336  # LLaDA mask token

    # --- Load dataset ---
    if rank == 0:
        print("Loading dataset...")

    from data.math import load_math500_dataset_and_reward

    dataloader, reward_fn = load_math500_dataset_and_reward(
        local_path="HuggingFaceH4/MATH-500",
        split="test",
        batch_size=config.batch_size_per_device,
        num_workers=4,
        max_level=config.max_level,
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
        if rank == 0:
            print(config.resume_ckpt)

        local_resume_path = config.resume_ckpt
        if os.path.exists(local_resume_path):
            if rank == 0:
                print(f"Resuming from {local_resume_path}")
            accelerator.load_state(local_resume_path)

        match = re.search(r"(\d+)$", config.resume_ckpt.rstrip("/"))
        if match:
            start_step = int(match.group(1))
            if rank == 0:
                print(f"start_step is {start_step}")

    dataloader_iter = iter(dataloader)

    if start_step > 0:
        if rank == 0:
            print(f"Skipping {start_step} batches...")
        for _ in range(start_step):
            next(dataloader_iter)

    # --- Output directory ---
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)

    group_size = config.num_generations * config.repeat_times * config.sample_repeat_times

    # --- Training loop setup ---
    if rank == 0:
        print(f"Starting training for {config.total_steps} steps...")
        print(f"Group size: {group_size}")
        print(f"Grad accumulation: {config.grad_accumulation}")
        print(f"Effective batch: {config.batch_size_per_device * dist.get_world_size() * config.grad_accumulation}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Temperature: {config.temperature}")
        print(f"Block size: {config.block_size}")
        print(f"Max level: {config.max_level}")
        print(f"Gain: {config.gain}")
        print(f"Scale by grad accumulation: {config.scale_by_grad_accum}")
        print(f"Max grad norm: {config.max_grad_norm}")

    for step in range(start_step, config.total_steps):
        optimizer.zero_grad(set_to_none=True)

        all_rewards = []

        for accum_idx in range(config.grad_accumulation):
            if rank == 0:
                print(
                    f"[Step {step + 1}/{config.total_steps}] "
                    f"[Accum {accum_idx + 1}/{config.grad_accumulation}] Sampling..."
                )

            with dist.ddp_sync(model, sync=(accum_idx == config.grad_accumulation - 1)):
                model.eval()

                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    # --- Rollout ---
                    batch = next(dataloader_iter)
                    inputs_chunks = []

                    if rank == 0:
                        print(f"use temperature: {config.temperature}")

                    for _ in range(config.repeat_times):
                        inputs = sample_with_repeat(
                            model=model,
                            batch=batch,
                            tokenizer=tokenizer,
                            device=device,
                            reward_fn=reward_fn,
                            temperature=config.temperature,
                            num_generations=config.num_generations,
                            steps=config.gen_steps,
                            gen_length=config.gen_length,
                            repeat_time=config.sample_repeat_times,
                            block_size=config.block_size,
                            apply_chat_template=True,
                        )
                        inputs_chunks.append(inputs)
                        torch.cuda.empty_cache()

                    # --- Compute Advantages ---
                    rewards = torch.cat([chunk["rewards"] for chunk in inputs_chunks], dim=0)

                    advantages = compute_group_advantages(
                        rewards,
                        group_size,
                    )

                    valid_samples = (advantages != 0).sum()

                    split_advantages = advantages.split(
                        config.num_generations * config.sample_repeat_times,
                        dim=0,
                    )

                    for chunk, adv in zip(inputs_chunks, split_advantages):
                        chunk["advantages"] = adv

                    accelerator.wait_for_everyone()

                    # --- Compute Loss ---
                    if rank == 0:
                        print(
                            f"[Step {step + 1}/{config.total_steps}] "
                            f"[Accum {accum_idx + 1}/{config.grad_accumulation}] Computing loss..."
                        )

                    model.train()

                    # Accumulation scaling:
                    # old behavior: gain=1.0
                    # new default: gain=1.0 / grad_accumulation
                    if config.scale_by_grad_accum:
                        effective_gain = config.gain / max(float(config.grad_accumulation), 1.0)
                    else:
                        effective_gain = config.gain

                    for inputs in inputs_chunks:
                        logprob_loss(
                            model=model,
                            inputs=inputs,
                            valid_samples=valid_samples,
                            gain=effective_gain,
                            temperature=config.temperature,
                            accelerator=accelerator,
                            gen_length=config.gen_length,
                        )
                        all_rewards.append(inputs["rewards"].detach())

                accelerator.wait_for_everyone()

                # Clear memory.
                for chunk in inputs_chunks:
                    for key in list(chunk.keys()):
                        del chunk[key]
                del inputs_chunks
                torch.cuda.empty_cache()

        # --- Grad Clip & Optimizer Step ---
        for param in model.parameters():
            if param.grad is not None:
                # Keep your original safeguard.
                param.grad = param.grad.float()
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

        # clip_grad_norm_ returns the norm BEFORE clipping.
        grad_norm = accelerator.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )

        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()

        # Compute norm AFTER clipping.
        grad_norm_clip = get_global_grad_norm_after_clip(
            model=model,
            accelerator=accelerator,
        )

        if hasattr(grad_norm_clip, "item"):
            grad_norm_clip = grad_norm_clip.item()

        optimizer.step()

        # --- Logging ---
        if (step + 1) % config.log_every == 0:
            all_rewards_tensor = torch.cat(all_rewards, dim=0)
            gathered_rewards = accelerator.gather(all_rewards_tensor)
            mean_reward = gathered_rewards.mean().item()

            if rank == 0:
                print(
                    f"[Step {step + 1}/{config.total_steps}] "
                    f"reward={mean_reward:.4f}, "
                    f"grad_norm={grad_norm:.4f}, "
                    f"grad_norm_clip={grad_norm_clip:.4f}"
                )

        # --- Save checkpoint ---
        if (step + 1) % config.save_every == 0:
            state_dict = accelerator.get_state_dict(model)

            if (step + 1) == config.total_steps:
                save_path = os.path.join(config.output_dir, f"training-state-{step + 1:06d}")
                accelerator.save_state(save_path)

            if rank == 0:
                save_path = os.path.join(config.output_dir, f"ckpt-{step + 1:06d}")
                accelerator.unwrap_model(model).save_pretrained(
                    save_path,
                    state_dict=state_dict,
                    safe_serialization=True,
                )
                print(f"Saved checkpoint to {save_path}")

        accelerator.wait_for_everyone()

    if rank == 0:
        print("\nTraining complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="JustGRPO Training")

    parser.add_argument("--run_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume checkpoint path")

    parser.add_argument("--temperature", type=float, default=1.0, help="rollout temperature")
    parser.add_argument("--lr", type=float, default=5e-6, help="lr")
    parser.add_argument("--block_size", type=int, default=1, help="Generate Block Size")
    parser.add_argument("--only_rollout", type=int, default=0)
    parser.add_argument("--max_level", type=int, default=3, help="Maximum difficulty level to train on")
    parser.add_argument("--total_steps", type=int, default=50, help="Total training steps")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N steps")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct",
        help="Path to pretrained model",
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Global RL loss gain before accumulation scaling.",
    )

    parser.add_argument(
        "--no_scale_by_grad_accum",
        action="store_true",
        help="Disable division by grad_accumulation in loss scaling.",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = TrainConfig(
        output_dir=args.run_dir,
        grad_accumulation=args.grad_accum,
        resume_ckpt=args.resume_ckpt,
        temperature=args.temperature,
        learning_rate=args.lr,
        block_size=args.block_size,
        only_rollout=args.only_rollout,
        max_level=args.max_level,
        total_steps=args.total_steps,
        save_every=args.save_every,
        model_path=args.model_path,
        gain=args.gain,
        scale_by_grad_accum=not args.no_scale_by_grad_accum,
        max_grad_norm=args.max_grad_norm,
    )

    train(config)