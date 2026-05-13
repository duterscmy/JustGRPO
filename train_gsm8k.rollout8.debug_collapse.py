import os
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import utils.distributed as dist
from grpo import sample_with_repeat, logprob_loss, compute_group_advantages


@dataclass
class TrainConfig:
    """Training hyperparameters for GRPO."""

    # --- Model ---
    model_path: str = "/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct"

    # --- Training ---
    batch_size_per_device: int = 1
    grad_accumulation: int = 8
    total_steps: int = 50
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
    temperature: float = 1.0

    gain: float = 1.0
    scale_by_grad_accum: bool = True
    advantage_clip: Optional[float] = None

    # --- Diagnostics ---
    log_policy_shift: bool = True
    policy_shift_stride: int = 8
    log_group_stats: bool = True

    # --- Token ---
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

    Used only for policy-shift diagnostics:
        old_logps: before optimizer.step()
        new_logps: after optimizer.step()

    Returns:
        logps: Tensor [batch_size, num_positions].
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


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return x.quantile(q).item()


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return x.mean().item()


def compute_policy_shift_stats(
    old_logps,
    new_logps,
    advantages=None,
    accelerator=None,
) -> Dict[str, float]:
    """
    Compute policy shift statistics.

    old_logps, new_logps:
        Tensor [B, P], where P is sampled token positions.

    advantages:
        Optional Tensor [B]. If provided, split policy shift by positive and
        negative advantage samples.
    """
    delta = (new_logps - old_logps).detach().float()
    ratio = torch.exp(delta).clamp(min=1e-6, max=1e6)

    flat_delta = delta.reshape(-1)
    flat_ratio = ratio.reshape(-1)

    pos_mask = None
    neg_mask = None

    if advantages is not None:
        adv = advantages.detach().float().view(-1, 1).expand_as(delta)
        pos_mask = (adv > 0).reshape(-1)
        neg_mask = (adv < 0).reshape(-1)

    if accelerator is not None:
        flat_delta = accelerator.gather(flat_delta)
        flat_ratio = accelerator.gather(flat_ratio)

        if pos_mask is not None:
            pos_mask = accelerator.gather(pos_mask.float()).bool()
            neg_mask = accelerator.gather(neg_mask.float()).bool()

    abs_delta = flat_delta.abs()

    stats = {
        "delta_logp_mean": flat_delta.mean().item(),
        "delta_logp_abs_mean": abs_delta.mean().item(),
        "delta_logp_p90": abs_delta.quantile(0.90).item(),
        "delta_logp_p99": abs_delta.quantile(0.99).item(),

        "ratio_mean": flat_ratio.mean().item(),
        "ratio_p01": flat_ratio.quantile(0.01).item(),
        "ratio_p05": flat_ratio.quantile(0.05).item(),
        "ratio_p10": flat_ratio.quantile(0.10).item(),
        "ratio_p50": flat_ratio.quantile(0.50).item(),
        "ratio_p90": flat_ratio.quantile(0.90).item(),
        "ratio_p99": flat_ratio.quantile(0.99).item(),

        "frac_ratio_gt_1p2": (flat_ratio > 1.2).float().mean().item(),
        "frac_ratio_lt_0p8": (flat_ratio < 0.8).float().mean().item(),
    }

    if pos_mask is not None:
        pos_delta = flat_delta[pos_mask]
        pos_ratio = flat_ratio[pos_mask]

        neg_delta = flat_delta[neg_mask]
        neg_ratio = flat_ratio[neg_mask]

        stats.update(
            {
                "pos_token_frac": pos_mask.float().mean().item(),
                "neg_token_frac": neg_mask.float().mean().item(),

                "pos_delta_mean": _safe_mean(pos_delta),
                "pos_delta_abs_mean": _safe_mean(pos_delta.abs()),
                "pos_ratio_p50": _safe_quantile(pos_ratio, 0.50),
                "pos_ratio_p90": _safe_quantile(pos_ratio, 0.90),
                "pos_frac_gt_1p2": (
                    (pos_ratio > 1.2).float().mean().item()
                    if pos_ratio.numel() > 0 else float("nan")
                ),

                "neg_delta_mean": _safe_mean(neg_delta),
                "neg_delta_abs_mean": _safe_mean(neg_delta.abs()),
                "neg_ratio_p10": _safe_quantile(neg_ratio, 0.10),
                "neg_ratio_p50": _safe_quantile(neg_ratio, 0.50),
                "neg_frac_lt_0p8": (
                    (neg_ratio < 0.8).float().mean().item()
                    if neg_ratio.numel() > 0 else float("nan")
                ),
            }
        )

    return stats


def compute_group_diagnostic_stats(
    rewards,
    advantages,
    group_size,
    accelerator=None,
) -> Dict[str, Any]:
    """
    Compute reward / advantage group diagnostics for one micro-batch.

    The layout follows compute_group_advantages:
        rewards.view(group_size, -1)
    so each column corresponds to one query group.
    """
    with torch.no_grad():
        rewards = rewards.detach().float()
        advantages = advantages.detach().float()

        grouped_rewards = rewards.view(group_size, -1).T.contiguous()      # [num_groups, group_size]
        grouped_adv = advantages.view(group_size, -1).T.contiguous()       # [num_groups, group_size]

        group_std = grouped_rewards.std(dim=1)
        valid_group = group_std > 1e-4

        pos_count = (grouped_rewards > 0.5).sum(dim=1).long()
        pos_hist = torch.bincount(pos_count, minlength=group_size + 1).float()

        adv_abs = grouped_adv.abs().reshape(-1)
        nonzero_adv = (grouped_adv.abs() > 1e-8).float()

        scalars = {
            "valid_group_ratio": valid_group.float().mean(),
            "reward_group_std_mean": group_std.mean(),
            "pos_count_mean": pos_count.float().mean(),
            "adv_abs_mean": adv_abs.mean(),
            "adv_abs_p90": adv_abs.quantile(0.90),
            "adv_abs_max": adv_abs.max(),
            "nonzero_adv_ratio": nonzero_adv.mean(),
            "num_groups": torch.tensor(float(grouped_rewards.shape[0]), device=rewards.device),
        }

        if accelerator is not None:
            gathered_scalars = {}
            for k, v in scalars.items():
                gv = accelerator.gather(v.reshape(1)).float()
                gathered_scalars[k] = gv.mean().item()

            gathered_hist = accelerator.gather(pos_hist)
            gathered_hist = gathered_hist.view(-1, group_size + 1).sum(dim=0)

            stats = gathered_scalars
            stats["pos_hist"] = gathered_hist.detach().cpu().tolist()
        else:
            stats = {k: v.item() for k, v in scalars.items()}
            stats["pos_hist"] = pos_hist.detach().cpu().tolist()

    return stats


def aggregate_micro_stats(stats_list: List[Dict[str, Any]], group_size: int) -> Dict[str, Any]:
    """
    Aggregate diagnostics across grad accumulation micro-batches.
    """
    if len(stats_list) == 0:
        return {}

    out = {}

    scalar_keys = [
        k for k in stats_list[0].keys()
        if k != "pos_hist"
    ]

    for k in scalar_keys:
        vals = [s[k] for s in stats_list if k in s]
        out[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    hist = np.zeros(group_size + 1, dtype=np.float64)
    for s in stats_list:
        if "pos_hist" in s:
            hist += np.array(s["pos_hist"], dtype=np.float64)

    out["pos_hist"] = hist.tolist()

    return out


def format_pos_hist(pos_hist: List[float]) -> str:
    parts = []
    for i, v in enumerate(pos_hist):
        if v > 0:
            parts.append(f"{i}:{int(v)}")
    return "{" + ",".join(parts) + "}"


def train(config: TrainConfig):
    """
    Main GRPO training loop.
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
    tokenizer.pad_token_id = config.mask_id

    # --- Load dataset ---
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

    group_size = config.num_generations * config.repeat_times * config.sample_repeat_times

    # --- Training setup log ---
    if rank == 0:
        print(f"Starting training for {config.total_steps} steps...")
        print(f"Group size: {group_size}")
        print(f"Grad accumulation: {config.grad_accumulation}")
        print(f"Effective batch: {config.batch_size_per_device * dist.get_world_size() * config.grad_accumulation}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Temperature: {config.temperature}")
        print(f"Block size: {config.block_size}")
        print(f"Gain: {config.gain}")
        print(f"Scale by grad accumulation: {config.scale_by_grad_accum}")
        print(f"Advantage clip: {config.advantage_clip}")
        print(f"Policy shift logging: {config.log_policy_shift}, stride={config.policy_shift_stride}")
        print(f"Group stats logging: {config.log_group_stats}")

    for step in range(start_step, config.total_steps):
        optimizer.zero_grad(set_to_none=True)

        all_rewards = []
        micro_group_stats = []

        # Monitor one rollout micro-batch per optimizer step.
        monitor_generated_ids = None
        monitor_prompt_len = None
        monitor_old_logps = None
        monitor_advantages = None
        monitor_rewards = None

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

                    advantages = compute_group_advantages(rewards, group_size)

                    if config.advantage_clip is not None:
                        advantages = advantages.clamp(-config.advantage_clip, config.advantage_clip)

                    if rank == 0:
                        print(f"advantages size: {advantages.size()}")

                    valid_samples = (advantages != 0).sum()

                    if config.log_group_stats:
                        group_stats = compute_group_diagnostic_stats(
                            rewards=rewards,
                            advantages=advantages,
                            group_size=group_size,
                            accelerator=accelerator,
                        )
                        micro_group_stats.append(group_stats)

                    split_advantages = advantages.split(
                        config.num_generations * config.sample_repeat_times,
                        dim=0,
                    )

                    for chunk, adv in zip(inputs_chunks, split_advantages):
                        chunk["advantages"] = adv

                    accelerator.wait_for_everyone()

                    # --- Record old log-probs before optimizer update ---
                    if (
                        config.log_policy_shift
                        and accum_idx == 0
                        and monitor_generated_ids is None
                        and len(inputs_chunks) > 0
                    ):
                        monitor_inputs = inputs_chunks[0]

                        monitor_generated_ids = monitor_inputs["generated_ids"].detach().clone()
                        monitor_prompt_len = monitor_inputs["prompt_len"]
                        monitor_advantages = monitor_inputs["advantages"].detach().clone()
                        monitor_rewards = monitor_inputs["rewards"].detach().clone()

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
                            gain=config.gain,
                            accelerator=accelerator,
                            gen_length=config.gen_length,
                            temperature=config.temperature,
                            mask_id=config.mask_id,
                            grad_accumulation=config.grad_accumulation,
                            scale_by_grad_accum=config.scale_by_grad_accum,
                            advantage_clip=config.advantage_clip,
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
                advantages=monitor_advantages,
                accelerator=accelerator,
            )

            del monitor_generated_ids
            del monitor_old_logps
            del monitor_new_logps
            del monitor_advantages
            del monitor_rewards
            torch.cuda.empty_cache()

        # --- Logging ---
        if (step + 1) % config.log_every == 0:
            all_rewards_tensor = torch.cat(all_rewards, dim=0)
            gathered_rewards = accelerator.gather(all_rewards_tensor)
            mean_reward = gathered_rewards.mean().item()

            group_stats_agg = (
                aggregate_micro_stats(micro_group_stats, group_size)
                if config.log_group_stats else {}
            )

            if rank == 0:
                msg = (
                    f"[Step {step + 1}/{config.total_steps}] "
                    f"reward={mean_reward:.4f}, "
                    f"grad={grad_norm:.4f}"
                )

                if policy_shift_stats is not None:
                    msg += (
                        f", dlogp_abs={policy_shift_stats['delta_logp_abs_mean']:.4f}"
                        f", dlogp_p90={policy_shift_stats['delta_logp_p90']:.4f}"
                        f", dlogp_p99={policy_shift_stats['delta_logp_p99']:.4f}"
                        f", ratio_p01={policy_shift_stats['ratio_p01']:.3f}"
                        f", ratio_p05={policy_shift_stats['ratio_p05']:.3f}"
                        f", ratio_p10={policy_shift_stats['ratio_p10']:.3f}"
                        f", ratio_p90={policy_shift_stats['ratio_p90']:.3f}"
                        f", ratio_p99={policy_shift_stats['ratio_p99']:.3f}"
                        f", frac>1.2={policy_shift_stats['frac_ratio_gt_1p2']:.3f}"
                        f", frac<0.8={policy_shift_stats['frac_ratio_lt_0p8']:.3f}"
                    )

                    if "pos_delta_mean" in policy_shift_stats:
                        msg += (
                            f", pos_dmean={policy_shift_stats['pos_delta_mean']:.4f}"
                            f", pos_frac>1.2={policy_shift_stats['pos_frac_gt_1p2']:.3f}"
                            f", neg_dmean={policy_shift_stats['neg_delta_mean']:.4f}"
                            f", neg_frac<0.8={policy_shift_stats['neg_frac_lt_0p8']:.3f}"
                        )

                if group_stats_agg:
                    msg += (
                        f", valid_group={group_stats_agg['valid_group_ratio']:.3f}"
                        f", nonzero_adv={group_stats_agg['nonzero_adv_ratio']:.3f}"
                        f", pos_count_mean={group_stats_agg['pos_count_mean']:.3f}"
                        f", adv_abs_mean={group_stats_agg['adv_abs_mean']:.3f}"
                        f", adv_abs_p90={group_stats_agg['adv_abs_p90']:.3f}"
                        f", adv_abs_max={group_stats_agg['adv_abs_max']:.3f}"
                        f", pos_hist={format_pos_hist(group_stats_agg['pos_hist'])}"
                    )

                print(msg)

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

    parser.add_argument("--block_size", type=int, default=1, help="Generate block size")
    parser.add_argument("--temperature", type=float, default=1.0, help="Rollout temperature")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--total_steps", type=int, default=50, help="Total training steps")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N steps")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct",
        help="Path to the model",
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Global loss gain.",
    )

    parser.add_argument(
        "--no_scale_by_grad_accum",
        action="store_true",
        help="Disable division by grad_accumulation in loss scale.",
    )

    parser.add_argument(
        "--advantage_clip",
        type=float,
        default=None,
        help="Clip advantages to [-value, value]. Default: no clipping.",
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

    parser.add_argument(
        "--no_group_stats_log",
        action="store_true",
        help="Disable reward/advantage group diagnostics.",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
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
        log_group_stats=not args.no_group_stats_log,
        gain=args.gain,
        scale_by_grad_accum=not args.no_scale_by_grad_accum,
        advantage_clip=args.advantage_clip,
        max_grad_norm=args.max_grad_norm,
    )

    train(config)