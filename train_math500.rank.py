import os
import re
import json
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

import utils.distributed as dist
from grpo import sample, sample_with_weighted_confidence,logprob_loss, compute_group_advantages



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

    # --- Gradient normalization ---
    gain: float = 1.0
    scale_by_grad_accum: bool = True

    # --- Dynamic sampling ---
    dynamic_sampling: bool = False
    dynamic_target_valid_groups: Optional[int] = None
    dynamic_max_attempts_per_group: int = 32

    # --- Misc ---
    output_dir: str = "./checkpoints_math500_num_generation{}".format(num_generations)
    log_every: int = 1
    save_every: int = 5
    resume_ckpt: Optional[str] = None

    only_rollout: int = 0


def load_training_progress(resume_ckpt: Optional[str]):
    """Load exact dataloader progress saved by this script."""
    if resume_ckpt is None:
        return None

    progress_path = os.path.join(resume_ckpt, "training_progress.json")
    if not os.path.isfile(progress_path):
        return None

    with open(progress_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_training_progress(save_path, completed_steps, data_batches_seen):
    """Atomically save optimizer-step and dataloader progress."""
    progress_path = os.path.join(save_path, "training_progress.json")
    tmp_path = progress_path + ".tmp"
    payload = {
        "completed_steps": int(completed_steps),
        "data_batches_seen": int(data_batches_seen),
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    os.replace(tmp_path, progress_path)


def distributed_min_int(value, accelerator, device):
    """Return the minimum integer value across all ranks."""
    value_tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    gathered = accelerator.gather(value_tensor)
    return int(gathered.min().item())


def distributed_sum_int(value, accelerator, device):
    """Return the sum of an integer value across all ranks."""
    value_tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    gathered = accelerator.gather(value_tensor)
    return int(gathered.sum().item())


def release_inputs_chunks(inputs_chunks):
    """Release tensors held by a sampled candidate group."""
    if inputs_chunks is None:
        return

    for chunk in inputs_chunks:
        for key in list(chunk.keys()):
            del chunk[key]


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
    print(f"Loading model from {config.model_path}...")
    from transformers import AutoTokenizer, AutoModel
    
    model = AutoModel.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.eval().to(device)

    # Activation checkpointing
    if hasattr(model, 'model') and hasattr(model.model, 'set_activation_checkpointing'):
        model.model.set_activation_checkpointing('whole_layer')
    
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token_id = 126336  # LLaDA mask token
    
    # --- Load dataset ---
    print("Loading dataset...")
    from data.math import load_math500_dataset_and_reward
    
    dataloader, reward_fn = load_math500_dataset_and_reward(
        local_path="HuggingFaceH4/MATH-500",
        split='test',
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
    data_batches_seen = 0
    saved_progress = load_training_progress(config.resume_ckpt)

    if config.resume_ckpt is not None:
        if rank == 0:
            print(config.resume_ckpt)

        local_resume_path = config.resume_ckpt
        if os.path.exists(local_resume_path):
            if rank == 0:
                print(f"Resuming from {local_resume_path}")
            accelerator.load_state(local_resume_path)

        if saved_progress is not None:
            start_step = int(saved_progress["completed_steps"])
            data_batches_seen = int(saved_progress["data_batches_seen"])
        else:
            match = re.search(r"(\d+)$", config.resume_ckpt.rstrip("/"))
            if match:
                start_step = int(match.group(1))

            # Backward-compatible fallback for old fixed-sampling checkpoints.
            data_batches_seen = start_step * config.grad_accumulation
            if config.dynamic_sampling:
                raise RuntimeError(
                    "Dynamic-sampling resume requires training_progress.json. "
                    "The old checkpoint does not contain the exact number of "
                    "sampled batches."
                )

        if rank == 0:
            print(f"start_step is {start_step}")
            print(f"data_batches_seen is {data_batches_seen}")

    dataloader_iter = iter(dataloader)

    if data_batches_seen > 0:
        if rank == 0:
            print(f"Skipping {data_batches_seen} previously consumed batches...")
        for _ in range(data_batches_seen):
            next(dataloader_iter)

    # --- Output directory ---
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)

    group_size = (
        config.num_generations
        * config.repeat_times
        * config.sample_repeat_times
    )
    update_group_count = (
        config.dynamic_target_valid_groups
        if config.dynamic_sampling
        and config.dynamic_target_valid_groups is not None
        else config.grad_accumulation
    )

    if update_group_count <= 0:
        raise ValueError("The number of update groups must be greater than zero.")
    if config.dynamic_max_attempts_per_group <= 0:
        raise ValueError("dynamic_max_attempts_per_group must be greater than zero.")

    # --- Training loop ---
    if rank == 0:
        print(f"Starting training for {config.total_steps} steps...")
        print(f"Group size: {group_size}")
        print(f"Grad accumulation: {config.grad_accumulation}")
        print(f"Update groups per rank: {update_group_count}")
        print(
            "Effective prompt groups: "
            f"{config.batch_size_per_device * dist.get_world_size() * update_group_count}"
        )
        print(f"Learning rate: {config.learning_rate}")
        print(f"Scale by update groups: {config.scale_by_grad_accum}")
        print(f"Dynamic sampling: {config.dynamic_sampling}")

    for step in range(start_step, config.total_steps):
        optimizer.zero_grad(set_to_none=True)

        all_rewards = []
        sampled_groups_local = 0
        accepted_groups_local = 0
        rejected_groups_local = 0
        extra_groups_local = 0

        for accum_idx in range(update_group_count):
            selected_inputs_chunks = None
            selected_valid_samples = None
            max_attempts = (
                config.dynamic_max_attempts_per_group
                if config.dynamic_sampling else 1
            )

            # All ranks keep sampling synchronously until each rank has one
            # non-zero-advantage group for this update slot. A rank that finds
            # one early keeps it, but still participates in rollout forwards
            # so FSDP collectives remain aligned.
            for attempt_idx in range(max_attempts):
                if rank == 0:
                    print(
                        f"[Step {step + 1}/{config.total_steps}] "
                        f"[Group {accum_idx + 1}/{update_group_count}] "
                        f"[Attempt {attempt_idx + 1}/{max_attempts}] Sampling..."
                    )

                model.eval()
                with torch.autocast(
                    device_type="cuda", enabled=True, dtype=torch.bfloat16
                ):
                    batch = next(dataloader_iter)
                    data_batches_seen += 1
                    sampled_groups_local += 1
                    candidate_inputs_chunks = []

                    for _ in range(config.repeat_times):
                        inputs = sample_with_weighted_confidence(
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
                        candidate_inputs_chunks.append(inputs)

                    rewards = torch.cat(
                        [chunk["rewards"] for chunk in candidate_inputs_chunks],
                        dim=0,
                    )
                    advantages = compute_group_advantages(rewards, group_size)
                    valid_samples = (advantages.abs() > 1e-8).sum()
                    is_valid_group = bool(valid_samples.item() > 0)

                    split_advantages = advantages.split(
                        config.num_generations * config.sample_repeat_times,
                        dim=0,
                    )
                    for chunk, adv in zip(
                        candidate_inputs_chunks, split_advantages
                    ):
                        chunk["advantages"] = adv

                if selected_inputs_chunks is None and (
                    is_valid_group or not config.dynamic_sampling
                ):
                    selected_inputs_chunks = candidate_inputs_chunks
                    selected_valid_samples = valid_samples
                    accepted_groups_local += 1
                else:
                    if is_valid_group:
                        extra_groups_local += 1
                    else:
                        rejected_groups_local += 1
                    release_inputs_chunks(candidate_inputs_chunks)

                local_ready = selected_inputs_chunks is not None
                all_ranks_ready = distributed_min_int(
                    int(local_ready), accelerator, device
                ) == 1
                torch.cuda.empty_cache()

                if all_ranks_ready:
                    break

            if not all_ranks_ready:
                release_inputs_chunks(selected_inputs_chunks)
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "Dynamic sampling could not find one valid group on every "
                    f"rank within {max_attempts} attempts for update group "
                    f"{accum_idx + 1}. Increase "
                    "--dynamic_max_attempts_per_group."
                )

            all_rewards.extend(
                chunk["rewards"].detach() for chunk in selected_inputs_chunks
            )

            with dist.ddp_sync(
                model, sync=(accum_idx == update_group_count - 1)
            ):
                if rank == 0:
                    print(
                        f"[Step {step + 1}/{config.total_steps}] "
                        f"[Group {accum_idx + 1}/{update_group_count}] "
                        "Computing loss..."
                    )

                model.train()
                for inputs in selected_inputs_chunks:
                    logprob_loss(
                        model=model,
                        inputs=inputs,
                        valid_samples=selected_valid_samples,
                        gain=config.gain,
                        temperature=config.temperature,
                        accelerator=accelerator,
                        gen_length=config.gen_length,
                        grad_accumulation=update_group_count,
                        scale_by_grad_accum=config.scale_by_grad_accum,
                    )

            accelerator.wait_for_everyone()
            release_inputs_chunks(selected_inputs_chunks)
            torch.cuda.empty_cache()

        # --- Grad Clip & Optimizer Step ---
        local_nonfinite_grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                nonfinite_mask = ~torch.isfinite(param.grad)
                local_nonfinite_grad_count += int(nonfinite_mask.sum().item())
                torch.nan_to_num(
                    param.grad, nan=0, posinf=0, neginf=0, out=param.grad
                )

        nonfinite_grad_count = distributed_sum_int(
            local_nonfinite_grad_count, accelerator, device
        )
        grad_norm_before = accelerator.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        )
        if hasattr(grad_norm_before, "item"):
            grad_norm_before = grad_norm_before.item()

        clip_coef = min(
            1.0,
            config.max_grad_norm / (grad_norm_before + 1e-6),
        )
        grad_norm_after = grad_norm_before * clip_coef
        was_clipped = grad_norm_before > config.max_grad_norm

        optimizer.step()

        # --- Logging ---
        if (step + 1) % config.log_every == 0:
            gathered_rewards = accelerator.gather(torch.cat(all_rewards, dim=0))
            mean_reward = gathered_rewards.mean().item()

            sampled_groups = distributed_sum_int(
                sampled_groups_local, accelerator, device
            )
            accepted_groups = distributed_sum_int(
                accepted_groups_local, accelerator, device
            )
            rejected_groups = distributed_sum_int(
                rejected_groups_local, accelerator, device
            )
            extra_groups = distributed_sum_int(
                extra_groups_local, accelerator, device
            )

            if rank == 0:
                msg = (
                    f"[Step {step + 1}/{config.total_steps}] "
                    f"reward={mean_reward:.4f}, "
                    f"grad_norm={grad_norm_before:.4f}, "
                    f"grad_norm_clip={grad_norm_after:.4f}, "
                    f"clip_coef={clip_coef:.4f}, "
                    f"was_clipped={int(was_clipped)}, "
                    f"nonfinite_grad={nonfinite_grad_count}"
                )
                if config.dynamic_sampling:
                    msg += (
                        f", sampled_groups={sampled_groups}"
                        f", accepted_groups={accepted_groups}"
                        f", rejected_groups={rejected_groups}"
                        f", extra_valid_groups={extra_groups}"
                        f", used_accept_rate="
                        f"{accepted_groups / max(sampled_groups, 1):.3f}"
                    )
                print(msg)

        # --- Save checkpoint ---
        if (step + 1) % config.save_every == 0:
            state_dict = accelerator.get_state_dict(model)

            if (step + 1) == config.total_steps:
                training_state_path = os.path.join(
                    config.output_dir,
                    f"training-state-{step + 1:06d}",
                )
                accelerator.save_state(training_state_path)
                accelerator.wait_for_everyone()

                if rank == 0:
                    save_training_progress(
                        training_state_path,
                        completed_steps=step + 1,
                        data_batches_seen=data_batches_seen,
                    )
                accelerator.wait_for_everyone()

            if rank == 0:
                save_path = os.path.join(
                    config.output_dir, f"ckpt-{step + 1:06d}"
                )
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
    parser.add_argument("--temperature", type=float, default=1.0,  help="rollout temperature")
    parser.add_argument("--lr", type=float, default=5e-6,  help="lr")
    parser.add_argument("--block_size", type=int, default=1, help="Generate Block Size")
    parser.add_argument("--only_rollout", type=int, default=0)
    parser.add_argument("--max_level", type=int, default=3, help="Maximum difficulty level to train on")
    parser.add_argument("--total_steps", type=int, default=50, help="Total training steps")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N steps")
    parser.add_argument("--model_path", type=str, default="/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct", help="Path to pretrained model")
    parser.add_argument("--gain", type=float, default=1.0, help="Global loss gain")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    grad_scale_group = parser.add_mutually_exclusive_group()
    grad_scale_group.add_argument(
        "--scale_by_grad_accum",
        dest="scale_by_grad_accum",
        action="store_true",
        help="Average accumulated gradients by the number of update groups (default)",
    )
    grad_scale_group.add_argument(
        "--no_scale_by_grad_accum",
        dest="scale_by_grad_accum",
        action="store_false",
        help="Disable division by the number of update groups",
    )
    parser.set_defaults(scale_by_grad_accum=True)

    parser.add_argument(
        "--dynamic_sampling",
        action="store_true",
        help=(
            "Reject all-zero-advantage groups and keep sampling until every "
            "rank has a valid group for each update slot"
        ),
    )
    parser.add_argument(
        "--dynamic_target_valid_groups",
        type=int,
        default=None,
        help="Valid groups per rank and optimizer step; defaults to --grad_accum",
    )
    parser.add_argument(
        "--dynamic_max_attempts_per_group",
        type=int,
        default=32,
        help="Maximum synchronized attempts for each valid update group",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create config from CLI args
    config = TrainConfig(
        output_dir=args.run_dir,
        grad_accumulation=args.grad_accum,
        resume_ckpt=args.resume_ckpt,
        temperature= args.temperature,
        learning_rate= args.lr,
        block_size=args.block_size,
        only_rollout=args.only_rollout,
        max_level=args.max_level,
        total_steps=args.total_steps,
        save_every=args.save_every,
        model_path=args.model_path,
        gain=args.gain,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        scale_by_grad_accum=args.scale_by_grad_accum,
        dynamic_sampling=args.dynamic_sampling,
        dynamic_target_valid_groups=args.dynamic_target_valid_groups,
        dynamic_max_attempts_per_group=args.dynamic_max_attempts_per_group,
    )

    train(config)
