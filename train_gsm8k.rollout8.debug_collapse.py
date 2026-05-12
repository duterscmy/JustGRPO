import os
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

import utils.distributed as dist
from grpo import sample, sample_with_repeat, logprob_loss, compute_group_advantages


@dataclass
class TrainConfig:
    """Training hyperparameters for GRPO."""

    # --- Model ---
    model_path: str = "/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct"

    # --- Training ---
    batch_size_per_device: int = 1
    grad_accumulation: int = 8
    total_steps: int = 10
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 1234
    num_generations: int = 4
    repeat_times: int = 1
    sample_repeat_times: int = 2
    gen_steps: int = 256
    gen_length: int = 256
    block_size: int = 1
    temperature: float = 0.6

    # --- Policy shift logging ---
    log_policy_shift: bool = True
    policy_shift_stride: int = 8
    mask_id: int = 126336

    # --- Misc ---
    output_dir: str = "./checkpoints"
    log_every: int = 1
    save_every: int = 5
    resume_ckpt: Optional[str] = None


@torch.no_grad()
def compute_sampled_ar_logps(
    model,
    generated_ids,
    prompt_len,
    gen_length=256,
    temperature=1.0,
    mask_id=126336,
    stride=8,
):
    """
    Compute AR-factorized token log-probs on sampled token positions.

    This is used only for policy-shift diagnostics:
        old_logps: before optimizer.step()
        new_logps: after optimizer.step()

    Args:
        model: dLLM model.
        generated_ids: Tensor [B, prompt_len + gen_length].
        prompt_len: prompt length.
        gen_length: completion length.
        temperature: logit temperature.
        mask_id: LLaDA mask token id.
        stride: compute logprob every `stride` tokens to reduce overhead.

    Returns:
        logps: Tensor [B, num_positions].
    """
    model.eval()

    batch_size = generated_ids.shape[0]
    device = generated_ids.device

    prompt_ids = generated_ids[:, :prompt_len]
    completion_ids = generated_ids[:, prompt_len:prompt_len + gen_length]

    logps = []
    positions = list(range(0, gen_length, stride))

    for t in positions:
        x = torch.cat(
            [
                prompt_ids,
                completion_ids[:, :t],
                torch.full(
                    (batch_size, gen_length - t),
                    mask_id,
                    device=device,
                    dtype=generated_ids.dtype,
                ),
            ],
            dim=1,
        )

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            logits = model(x).logits / temperature

        log_prob = F.log_softmax(
            logits[:, prompt_len + t, :].float(),
            dim=-1,
        )

        token_log_prob = log_prob.gather(
            -1,
            completion_ids[:, t:t + 1],
        ).squeeze(-1)

        logps.append(token_log_prob)

    return torch.stack(logps, dim=1)


def compute_policy_shift_stats(old_logps, new_logps, accelerator=None):
    """
    Compute policy shift statistics from old/new sampled token log-probs.

    Args:
        old_logps: Tensor [B, num_positions], before optimizer.step().
        new_logps: Tensor [B, num_positions], after optimizer.step().
        accelerator: accelerate accelerator, used to gather across GPUs.

    Returns:
        dict of scalar python floats.
    """
    delta = (new_logps - old_logps).detach().float().reshape(-1)

    # Gather token-level deltas across all GPUs.
    if accelerator is not None:
        delta = accelerator.gather(delta)

    ratio = torch.exp(delta).clamp(max=1e6)

    stats = {
        "delta_logp_mean": delta.mean().item(),
        "delta_logp_abs_mean": delta.abs().mean().item(),
        "delta_logp_p90": delta.abs().quantile(0.90).item(),
        "delta_logp_p99": delta.abs().quantile(0.99).item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_p90": ratio.quantile(0.90).item(),
        "ratio_p99": ratio.quantile(0.99).item(),
        "frac_ratio_gt_1p2": (ratio > 1.2).float().mean().item(),
        "frac_ratio_lt_0p8": (ratio < 0.8).float().mean().item(),
    }

    return stats


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
    tokenizer.pad_token_id = config.mask_id  # LLaDA mask token

    # --- Load dataset (GSM8K) ---
    if rank == 0:
        print("Loading GSM8K dataset...")

    from data.math import load_gsm8k_dataset_and_reward

    dataloader, reward_fn = load_gsm8k_dataset_and_reward(
        local_path="gsm8k",
        split="test",
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

    # --- Training loop ---
    if rank == 0:
        print(f"Starting training for {config.total_steps} steps...")
        print(f"Group size: {config.num_generations * config.repeat_times * config.sample_repeat_times}")
        print(f"Grad accumulation: {config.grad_accumulation}")
        print(f"Effective batch: {config.batch_size_per_device * dist.get_world_size() * config.grad_accumulation}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Policy shift logging: {config.log_policy_shift}, stride={config.policy_shift_stride}")

    for step in range(start_step, config.total_steps):
        optimizer.zero_grad(set_to_none=True)

        all_rewards = []

        # These are used to monitor one rollout batch per optimizer step.
        monitor_generated_ids = None
        monitor_prompt_len = None
        monitor_old_logps = None

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

                    for _ in range(config.repeat_times):
                        inputs = sample_with_repeat(
                            model=model,
                            batch=batch,
                            tokenizer=tokenizer,
                            device=device,
                            reward_fn=reward_fn,
                            num_generations=config.num_generations,
                            steps=config.gen_steps,
                            gen_length=config.gen_length,
                            repeat_time=config.sample_repeat_times,
                            block_size=config.block_size,
                            temperature=config.temperature,
                            apply_chat_template=True,
                        )
                        inputs_chunks.append(inputs)
                        torch.cuda.empty_cache()

                    # --- Compute Advantages ---
                    rewards = torch.cat([chunk["rewards"] for chunk in inputs_chunks], dim=0)

                    if rank == 0:
                        print(f"reward size: {rewards.size()}")

                    group_size = config.num_generations * config.repeat_times * config.sample_repeat_times
                    advantages = compute_group_advantages(rewards, group_size)

                    if rank == 0:
                        print(f"advantages size: {advantages.size()}")

                    valid_samples = (advantages != 0).sum()

                    split_advantages = advantages.split(
                        config.num_generations * config.sample_repeat_times,
                        dim=0,
                    )

                    for chunk, adv in zip(inputs_chunks, split_advantages):
                        chunk["advantages"] = adv

                    accelerator.wait_for_everyone()

                    # --- Record old log-probs before optimizer update ---
                    # We only monitor the first accumulation micro-batch to reduce overhead.
                    if (
                        config.log_policy_shift
                        and accum_idx == 0
                        and monitor_generated_ids is None
                        and len(inputs_chunks) > 0
                    ):
                        monitor_inputs = inputs_chunks[0]

                        monitor_generated_ids = monitor_inputs["generated_ids"].detach().clone()
                        monitor_prompt_len = monitor_inputs["prompt_len"]

                        monitor_old_logps = compute_sampled_ar_logps(
                            model=model,
                            generated_ids=monitor_generated_ids,
                            prompt_len=monitor_prompt_len,
                            gen_length=config.gen_length,
                            temperature=config.temperature,
                            mask_id=config.mask_id,
                            stride=config.policy_shift_stride,
                        ).detach()

                    # --- Compute Loss ---
                    if rank == 0:
                        print(
                            f"[Step {step + 1}/{config.total_steps}] "
                            f"[Accum {accum_idx + 1}/{config.grad_accumulation}] Computing loss..."
                        )

                    model.train()

                    for inputs in inputs_chunks:
                        logprob_loss(
                            model=model,
                            inputs=inputs,
                            valid_samples=valid_samples,
                            gain=1.0,
                            accelerator=accelerator,
                            gen_length=config.gen_length,
                            temperature=config.temperature,
                        )
                        all_rewards.append(inputs["rewards"].detach())

                accelerator.wait_for_everyone()

                # Clear all chunks, not only the last `inputs`.
                for chunk in inputs_chunks:
                    for key in list(chunk.keys()):
                        del chunk[key]
                del inputs_chunks
                torch.cuda.empty_cache()

        # --- Grad Clip & Optimizer Step ---
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

        grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()

        optimizer.step()

        # --- Policy shift monitor after optimizer update ---
        policy_shift_stats = None

        if config.log_policy_shift and monitor_generated_ids is not None and monitor_old_logps is not None:
            accelerator.wait_for_everyone()

            monitor_new_logps = compute_sampled_ar_logps(
                model=model,
                generated_ids=monitor_generated_ids,
                prompt_len=monitor_prompt_len,
                gen_length=config.gen_length,
                temperature=config.temperature,
                mask_id=config.mask_id,
                stride=config.policy_shift_stride,
            ).detach()

            policy_shift_stats = compute_policy_shift_stats(
                old_logps=monitor_old_logps,
                new_logps=monitor_new_logps,
                accelerator=accelerator,
            )

            del monitor_generated_ids
            del monitor_old_logps
            del monitor_new_logps
            torch.cuda.empty_cache()

        # --- Logging ---
        if (step + 1) % config.log_every == 0:
            all_rewards_tensor = torch.cat(all_rewards, dim=0)
            gathered_rewards = accelerator.gather(all_rewards_tensor)
            mean_reward = gathered_rewards.mean().item()

            if rank == 0:
                if policy_shift_stats is not None:
                    print(
                        f"[Step {step + 1}/{config.total_steps}] "
                        f"reward={mean_reward:.4f}, "
                        f"grad={grad_norm:.4f}, "
                        f"dlogp_abs={policy_shift_stats['delta_logp_abs_mean']:.4f}, "
                        f"dlogp_p90={policy_shift_stats['delta_logp_p90']:.4f}, "
                        f"dlogp_p99={policy_shift_stats['delta_logp_p99']:.4f}, "
                        f"ratio_p90={policy_shift_stats['ratio_p90']:.3f}, "
                        f"ratio_p99={policy_shift_stats['ßratio_p99']:.3f}, "
                        f"frac>1.2={policy_shift_stats['frac_ratio_gt_1p2']:.3f}, "
                        f"frac<0.8={policy_shift_stats['frac_ratio_lt_0p8']:.3f}"
                    )
                else:
                    print(
                        f"[Step {step + 1}/{config.total_steps}] "
                        f"reward={mean_reward:.4f}, grad={grad_norm:.4f}"
                    )

        # --- Save checkpoint ---
        if (step + 1) % config.save_every == 0:
            state_dict = accelerator.get_state_dict(model)

            if step + 1 == config.total_steps:
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
    parser.add_argument("--block_size", type=int, default=1, help="Generate Block Size")
    parser.add_argument("--temperature", type=float, default=1.0, help="rollout temperature")
    parser.add_argument("--lr", type=float, default=5e-6, help="lr")
    parser.add_argument("--total_steps", type=int, default=50, help="Total training steps")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N steps")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct",
        help="Path to the model",
    )

    parser.add_argument(
        "--policy_shift_stride",
        type=int,
        default=8,
        help="Stride for policy shift logprob monitoring. Larger is cheaper.",
    )

    parser.add_argument(
        "--no_policy_shift_log",
        action="store_true",
        help="Disable policy shift logging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = TrainConfig(
        output_dir=args.run_dir,
        grad_accumulation=args.grad_accum,
        resume_ckpt=args.resume_ckpt,
        block_size=args.block_size,
        temperature=args.temperature,
        learning_rate=args.lr,
        total_steps=args.total_steps,
        save_every=args.save_every,
        model_path=args.model_path,
        policy_shift_stride=args.policy_shift_stride,
        log_policy_shift=not args.no_policy_shift_log,
    )

    train(config)