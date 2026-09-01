"""GSM8K dTTRL with diversity-controlled rollout temperature.

This is an additive entry point: it reuses the current
``train_gsm8k.rollout8.rank.py`` training loop and does not edit it.  Rollout
confidence remains in the log, but the controller uses only the exact number
of math-distinct answers among the eight rollouts.

At the end of every optimizer step, diversity is averaged over every prompt
group and every distributed rank.  The resulting temperature is used from the
next optimizer step onward.  No group is rejected or re-sampled.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable, Sequence

import torch
import torch.distributed as torch_dist

import grpo
from data.math import extract_answer, math_equal
from diversity_temperature import DiversityTemperatureController
from utils.generate import generate_with_confidence


ROOT = Path(__file__).resolve().parent
BASE_TRAIN_SCRIPT = ROOT / "train_gsm8k.rollout8.rank.py"
MAJORITY_VOTE_SCRIPT = ROOT / "train_gsm8k.rollout8.majority_vote.py"


def load_local_module(module_name: str, path: Path) -> ModuleType:
    """Load a local Python file whose filename contains dots."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_answers(responses: Iterable[str]) -> list:
    """Map extracted answers to the same math-equivalence classes as voting."""
    representatives = []
    normalized = []

    for response in responses:
        answer = extract_answer(response)
        for representative in representatives:
            if math_equal(answer, representative):
                normalized.append(representative)
                break
        else:
            representatives.append(answer)
            normalized.append(answer)

    return normalized


def distinct_answer_count(normalized_answers: Sequence) -> int:
    """Count equivalence classes without requiring answers to be hashable."""
    representatives = []
    for answer in normalized_answers:
        if not any(math_equal(answer, item) for item in representatives):
            representatives.append(answer)
    return len(representatives)


class DistributedDiversityRuntime:
    """Average one optimizer step of diversity and update a shared controller."""

    def __init__(
        self,
        controller: DiversityTemperatureController,
        groups_per_step: int,
    ) -> None:
        if groups_per_step <= 0:
            raise ValueError("groups_per_step must be greater than zero")

        self.controller = controller
        self.groups_per_step = int(groups_per_step)
        self.calls_in_step = 0
        self.global_diversity_sum = 0.0
        self.global_group_count = 0.0
        self.control_step = 0
        self.printed_configuration = False

    @property
    def temperature(self) -> float:
        return self.controller.temperature

    @staticmethod
    def is_primary_process() -> bool:
        return (
            not torch_dist.is_available()
            or not torch_dist.is_initialized()
            or torch_dist.get_rank() == 0
        )

    def print_configuration(self) -> None:
        if self.printed_configuration:
            return
        self.printed_configuration = True

        if not self.is_primary_process():
            return

        print("Voting mode: uniform majority (confidence is logging-only)")
        print("Rollout mode: AR (block_size=1)")
        print("Dynamic sampling: disabled; no rollout is re-sampled")
        print("Adaptive rollout temperature: diversity-only")
        print(f"Diversity target: {self.controller.target_diversity}")
        print(f"Diversity EMA decay: {self.controller.ema_decay}")
        print(f"Diversity controller gain: {self.controller.gain}")
        print(f"Diversity deadband: +/-{self.controller.deadband}")
        print(
            "Temperature bounds: "
            f"[{self.controller.min_temperature}, "
            f"{self.controller.max_temperature}]"
        )
        print(
            "Maximum relative temperature change per optimizer step: "
            f"{self.controller.max_change}"
        )
        print(f"Sampled groups per rank and step: {self.groups_per_step}")

    def observe(self, distinct_count: int, device: torch.device) -> None:
        """Observe one local group; update after one full optimizer step."""
        if distinct_count <= 0:
            raise ValueError("distinct_count must be greater than zero")

        metrics = torch.tensor(
            [float(distinct_count), 1.0],
            device=device,
            dtype=torch.float32,
        )
        if torch_dist.is_available() and torch_dist.is_initialized():
            torch_dist.all_reduce(metrics, op=torch_dist.ReduceOp.SUM)

        diversity_sum, group_count = metrics.detach().cpu().tolist()
        self.global_diversity_sum += diversity_sum
        self.global_group_count += group_count
        self.calls_in_step += 1

        if self.calls_in_step < self.groups_per_step:
            return

        mean_diversity = (
            self.global_diversity_sum / max(self.global_group_count, 1.0)
        )
        update = self.controller.update(mean_diversity)
        self.control_step += 1

        if self.is_primary_process():
            print(
                "diversity_temperature| "
                f"step: {self.control_step} | "
                f"temperature: {update['old_temperature']:.4f} | "
                f"next_temperature: {update['new_temperature']:.4f} | "
                f"step_mean_diversity: {update['mean_diversity']:.4f} | "
                f"ema_diversity: {update['smoothed_diversity']:.4f} | "
                f"target_diversity: {self.controller.target_diversity:.4f} | "
                f"deadband: {self.controller.deadband:.4f} | "
                f"action: {update['action']} | "
                f"relative_change: "
                f"{update['applied_relative_change']:.4f} | "
                f"global_group_count: {int(self.global_group_count)}",
                flush=True,
            )

        self.calls_in_step = 0
        self.global_diversity_sum = 0.0
        self.global_group_count = 0.0


def build_diversity_sampler(
    runtime: DistributedDiversityRuntime,
    majority_vote_module: ModuleType,
):
    """Generate eight rollouts and preserve the existing analysis logs."""

    @torch.no_grad()
    def sample_with_diversity_temperature(
        model,
        batch,
        tokenizer,
        device,
        reward_fn=None,
        num_generations=1,
        temperature=1.0,
        steps=256,
        gen_length=256,
        repeat_time=1,
        block_size=1,
        apply_chat_template=True,
    ):
        del reward_fn, temperature

        if block_size != 1:
            raise ValueError(
                "This experiment is restricted to AR rollout: block_size must be 1."
            )
        if len(batch["problems"]) != 1:
            raise ValueError(
                "This entry point requires batch_size_per_device=1."
            )

        runtime.print_configuration()
        rollout_temperature = runtime.temperature
        print(f"adaptive_temperature_used: {rollout_temperature:.4f}")

        if apply_chat_template:
            prompts = tokenizer.apply_chat_template(
                [
                    [{"role": "user", "content": problem}]
                    for problem in batch["problems"]
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompts = batch["problems"]

        prompt_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )["input_ids"].to(device)

        generated_with_confidence = []
        print(f"=======block size:{block_size}======")

        for _ in range(repeat_time):
            repeated_prompt_ids = prompt_ids.repeat(num_generations, 1)
            generated_ids, confidence_values = generate_with_confidence(
                model=model,
                prompt=repeated_prompt_ids,
                steps=steps,
                gen_length=gen_length,
                temperature=rollout_temperature,
                block_length=block_size,
            )

            completion_lengths = grpo.get_completion_lengths(
                generated_ids,
                prompt_len=prompt_ids.shape[1],
            )
            print(f"avg_gen_length: {completion_lengths.float().mean():.1f}")
            generated_with_confidence.extend(
                zip(generated_ids, confidence_values)
            )

        if not generated_with_confidence:
            raise RuntimeError("Rollout generation returned no samples")

        all_generated_ids = torch.stack(
            [generated_id for generated_id, _ in generated_with_confidence]
        )
        all_confidences = [
            confidence for _, confidence in generated_with_confidence
        ]
        print("confidence list:", all_confidences)
        print(all_generated_ids.size())

        responses = tokenizer.batch_decode(
            all_generated_ids,
            skip_special_tokens=True,
        )
        rewards = majority_vote_module.reward_ttrl_uniform_with_confidence_log(
            batch=batch,
            responses=responses,
            num_generations=num_generations * repeat_time,
            device=device,
            confidences=all_confidences,
        ).float()

        normalized_answers = normalize_answers(responses)
        runtime.observe(
            distinct_count=distinct_answer_count(normalized_answers),
            device=device,
        )

        return {
            "generated_ids": all_generated_ids,
            "prompt_len": prompt_ids.shape[1],
            "rewards": rewards,
            "rollout_temperature": rollout_temperature,
        }

    return sample_with_diversity_temperature


def patch_loss_temperature(base_module: ModuleType) -> None:
    """Use the same temperature for generation and sampled-token log-probs."""
    original_logprob_loss = base_module.logprob_loss

    def logprob_loss_with_rollout_temperature(*args, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs is None and len(args) >= 2:
            inputs = args[1]
        if inputs is None or "rollout_temperature" not in inputs:
            raise KeyError("Sampled inputs are missing rollout_temperature")

        kwargs["temperature"] = float(inputs["rollout_temperature"])
        return original_logprob_loss(*args, **kwargs)

    base_module.logprob_loss = logprob_loss_with_rollout_temperature


def parse_controller_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diversity_temp_target", type=float, default=2.0)
    parser.add_argument("--diversity_temp_ema_decay", type=float, default=0.6)
    parser.add_argument("--diversity_temp_gain", type=float, default=0.5)
    parser.add_argument("--diversity_temp_deadband", type=float, default=0.1)
    parser.add_argument("--diversity_temp_max_change", type=float, default=0.10)
    parser.add_argument("--diversity_temp_min", type=float, default=0.3)
    parser.add_argument("--diversity_temp_max", type=float, default=1.5)
    return parser.parse_known_args()


def build_train_config(base_module: ModuleType, args: argparse.Namespace):
    """Mirror rank.py's current command-line-to-config mapping."""
    return base_module.TrainConfig(
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
        scale_by_grad_accum=args.scale_by_grad_accum,
        dynamic_sampling=args.dynamic_sampling,
        dynamic_target_valid_groups=args.dynamic_target_valid_groups,
        dynamic_max_attempts_per_group=args.dynamic_max_attempts_per_group,
        advantage_clip=args.advantage_clip,
        max_grad_norm=args.max_grad_norm,
        kl_beta=args.kl_beta,
        seed=args.seed,
    )


def main() -> None:
    controller_args, base_argv = parse_controller_args()

    if any(
        argument == "--dynamic_sampling"
        or argument.startswith("--dynamic_sampling=")
        for argument in base_argv
    ):
        raise SystemExit(
            "Diversity-controlled temperature does not re-sample groups; "
            "remove --dynamic_sampling."
        )

    base_module = load_local_module(
        "dttrl_gsm8k_rank_diversity_temperature",
        BASE_TRAIN_SCRIPT,
    )
    majority_vote_module = load_local_module(
        "dttrl_gsm8k_uniform_majority_diversity_temperature",
        MAJORITY_VOTE_SCRIPT,
    )

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *base_argv]
        base_args = base_module.parse_args()
    finally:
        sys.argv = original_argv

    if base_args.block_size != 1:
        raise SystemExit("This experiment requires --block_size 1 (AR rollout).")
    if base_args.dynamic_sampling:
        raise SystemExit("Dynamic Sampling must be disabled in this experiment.")

    controller = DiversityTemperatureController(
        temperature=base_args.temperature,
        target_diversity=controller_args.diversity_temp_target,
        ema_decay=controller_args.diversity_temp_ema_decay,
        gain=controller_args.diversity_temp_gain,
        deadband=controller_args.diversity_temp_deadband,
        max_change=controller_args.diversity_temp_max_change,
        min_temperature=controller_args.diversity_temp_min,
        max_temperature=controller_args.diversity_temp_max,
    )
    runtime = DistributedDiversityRuntime(
        controller=controller,
        groups_per_step=base_args.grad_accum,
    )

    base_module.sample_with_weighted_confidence = build_diversity_sampler(
        runtime=runtime,
        majority_vote_module=majority_vote_module,
    )
    patch_loss_temperature(base_module)

    config = build_train_config(base_module, base_args)
    base_module.train(config)


if __name__ == "__main__":
    main()
