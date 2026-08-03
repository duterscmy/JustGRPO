import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import utils.distributed as dist
from data.planning import load_planning_dataset_and_reward
from grpo import (
    compute_group_advantages,
    logprob_loss,
    sample_with_weighted_confidence,
)


@dataclass
class TrainConfig:
    task: str = "sudoku"
    data_dir: str = "dataset"
    model_path: str = "/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct"
    batch_size_per_device: int = 1
    grad_accumulation: int = 8
    total_steps: int = 30
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 1234
    num_generations: int = 4
    sample_repeat_times: int = 2
    gen_steps: int = 256
    gen_length: int = 256
    block_size: int = 32
    temperature: float = 1.0
    output_dir: str = "./checkpoints_planning"
    log_every: int = 1
    save_every: int = 5
    resume_ckpt: Optional[str] = None
    num_workers: int = 4

    @property
    def group_size(self) -> int:
        return self.num_generations * self.sample_repeat_times


def train(config: TrainConfig) -> None:
    dist.init()
    rank = dist.get_rank()
    device = torch.device("cuda")

    print("=" * 60)
    print(f"dTTRL training: {config.task}")
    print("=" * 60)

    np.random.seed((config.seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model from {config.model_path}")
    model = AutoModel.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
        model.model.set_activation_checkpointing("whole_layer")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 126336

    dataloader, reward_fn = load_planning_dataset_and_reward(
        task=config.task,
        data_dir=config.data_dir,
        batch_size=config.batch_size_per_device,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )

    accelerator = dist.get_accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    start_step = 0
    if config.resume_ckpt:
        if os.path.exists(config.resume_ckpt):
            print(f"Resuming from {config.resume_ckpt}")
            accelerator.load_state(config.resume_ckpt)
        match = re.search(r"(\d+)$", config.resume_ckpt.rstrip("/"))
        if match:
            start_step = int(match.group(1))

    dataloader_iter = iter(dataloader)
    for _ in range(start_step):
        next(dataloader_iter)

    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)

    print(f"steps={config.total_steps} group_size={config.group_size}")
    print(
        f"gen_length={config.gen_length} gen_steps={config.gen_steps} "
        f"block_size={config.block_size} temperature={config.temperature}"
    )

    for step in range(start_step, config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        all_rewards = []
        for accum_idx in range(config.grad_accumulation):
            print(
                f"[Step {step + 1}/{config.total_steps}] "
                f"[Accum {accum_idx + 1}/{config.grad_accumulation}] sampling"
            )
            with dist.ddp_sync(model, sync=(accum_idx == config.grad_accumulation - 1)):
                model.eval()
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    batch = next(dataloader_iter)
                    inputs = sample_with_weighted_confidence(
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
                    rewards = inputs["rewards"]
                    advantages = compute_group_advantages(rewards, config.group_size)
                    inputs["advantages"] = advantages
                    nonzero_advantages = (advantages != 0).sum()
                    # Every rank must execute backward under DDP/FSDP.  Clamping the
                    # denominator keeps an all-tied group as a valid zero-gradient
                    # update instead of causing division by zero or a rank mismatch.
                    valid_samples = nonzero_advantages.clamp_min(1)
                    all_rewards.append(rewards.detach())
                    if nonzero_advantages.item() == 0:
                        print("All group advantages are zero; applying a zero-gradient update")
                    model.train()
                    logprob_loss(
                        model=model,
                        inputs=inputs,
                        valid_samples=valid_samples,
                        gain=1.0,
                        accelerator=accelerator,
                        gen_length=config.gen_length,
                        temperature=config.temperature,
                    )

                accelerator.wait_for_everyone()
                for key in list(inputs):
                    del inputs[key]

        for parameter in model.parameters():
            if parameter.grad is not None:
                torch.nan_to_num(
                    parameter.grad, nan=0, posinf=0, neginf=0, out=parameter.grad
                )
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
        optimizer.step()

        if (step + 1) % config.log_every == 0:
            gathered = accelerator.gather(torch.cat(all_rewards, dim=0))
            print(
                f"[Step {step + 1}/{config.total_steps}] "
                f"reward={gathered.mean().item():.4f} grad={grad_norm:.4f}"
            )

        if (step + 1) % config.save_every == 0:
            state_dict = accelerator.get_state_dict(model)
            if step + 1 == config.total_steps:
                accelerator.save_state(
                    os.path.join(config.output_dir, f"training-state-{step + 1:06d}")
                )
            if rank == 0:
                save_path = os.path.join(config.output_dir, f"ckpt-{step + 1:06d}")
                accelerator.unwrap_model(model).save_pretrained(
                    save_path,
                    state_dict=state_dict,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
        accelerator.wait_for_everyone()

    print("Training complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dTTRL Sudoku/Countdown training")
    parser.add_argument("--task", choices=["sudoku", "countdown"], required=True)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--run_dir", default="./checkpoints_planning")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--resume_ckpt", default=None)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--total_steps", type=int, default=30)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--gen_steps", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--sample_repeat_times", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    temperature = args.temperature
    if temperature is None:
        temperature = 0.3 if args.task == "sudoku" else 1.0
    train(
        TrainConfig(
            task=args.task,
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.run_dir,
            grad_accumulation=args.grad_accum,
            resume_ckpt=args.resume_ckpt,
            block_size=args.block_size,
            temperature=temperature,
            learning_rate=args.lr,
            total_steps=args.total_steps,
            save_every=args.save_every,
            gen_length=args.gen_length,
            gen_steps=args.gen_steps,
            num_generations=args.num_generations,
            sample_repeat_times=args.sample_repeat_times,
            num_workers=args.num_workers,
        )
    )
