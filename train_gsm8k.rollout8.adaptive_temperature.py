"""GSM8K dTTRL with uniform voting and adaptive rollout temperature.

This additive entry point reuses ``train_gsm8k.rollout8.rank.py`` without
editing it.  It also reuses the uniform-majority reward and its existing log
format from ``train_gsm8k.rollout8.majority_vote.py``.

The schedule is intentionally small:

* the first metric window at the initial temperature becomes the target;
* confidence above the target raises temperature;
* confidence below the target lowers temperature;
* if confidence is not falling, reduced diversity plus fewer valid-gradient
  groups adds a small upward exploration correction.

The same controller is used for AR and Block Diffusion rollouts.  Temperature
is globally aggregated across ranks and is therefore identical on all GPUs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable, Sequence

import torch
import torch.distributed as torch_dist

import grpo
from adaptive_temperature import AdaptiveTemperatureController
from data.math import extract_answer, math_equal
from utils.generate import generate_with_confidence


ROOT = Path(__file__).resolve().parent
BASE_TRAIN_SCRIPT = ROOT / "train_gsm8k.rollout8.rank.py"
MAJORITY_VOTE_SCRIPT = ROOT / "train_gsm8k.rollout8.majority_vote.py"


def load_local_module(module_name: str, path: Path) -> ModuleType:
    """Load a Python file whose filename contains dots."""
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
    """Use the same math-equivalence classes as the voting reward."""
    extracted_answers = [extract_answer(response) for response in responses]
    representatives = []
    normalized_answers = []

    for answer in extracted_answers:
        for representative in representatives:
            if math_equal(answer, representative):
                normalized_answers.append(representative)
                break
        else:
            representatives.append(answer)
            normalized_answers.append(answer)

    return normalized_answers


def answer_distribution_metrics(normalized_answers: Sequence) -> tuple[float, float, float, float]:
    """Return effective diversity, valid-group flag, majority ratio and margin."""
    if not normalized_answers:
        return 0.0, 0.0, 0.0, 0.0

    counts = sorted(Counter(normalized_answers).values(), reverse=True)
    total = float(sum(counts))
    probabilities = [count / total for count in counts]
    answer_entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    effective_diversity = math.exp(answer_entropy)

    top1 = counts[0]
    top2 = counts[1] if len(counts) > 1 else 0
    valid_group = float(len(counts) > 1)
    majority_ratio = top1 / total
    vote_margin = (top1 - top2) / total
    return effective_diversity, valid_group, majority_ratio, vote_margin


class DistributedTemperatureRuntime:
    """Aggregate rollout metrics and update one shared temperature."""

    def __init__(
        self,
        controller: AdaptiveTemperatureController,
        window_groups: int,
    ) -> None:
        if window_groups <= 0:
            raise ValueError("window_groups must be greater than zero")

        self.controller = controller
        self.window_groups = int(window_groups)
        self.window_index = 0
        self.calls_in_window = 0
        self.metric_sums = [0.0] * 7
        self.printed_configuration = False

    @property
    def temperature(self) -> float:
        return self.controller.temperature

    def print_configuration(self) -> None:
        if self.printed_configuration:
            return

        print(
            "Voting mode: uniform majority "
            "(confidence is not used as a vote weight)"
        )
        print("Dynamic sampling: forced off by this entry point")
        print("Adaptive rollout temperature: enabled")
        print(f"Adaptive initial temperature: {self.controller.temperature}")
        print(
            "Adaptive temperature bounds: "
            f"[{self.controller.min_temperature}, "
            f"{self.controller.max_temperature}]"
        )
        print(f"Adaptive metric window: {self.window_groups} sampled groups per rank")
        print(f"Adaptive confidence gain: {self.controller.confidence_gain}")
        print(f"Adaptive valid-group gain: {self.controller.valid_group_gain}")
        print(f"Adaptive confidence deadband: {self.controller.confidence_deadband}")
        print(f"Adaptive max change per window: {self.controller.max_change}")
        self.printed_configuration = True

    def observe(
        self,
        confidences: Sequence[float],
        normalized_answers: Sequence,
        device: torch.device,
    ) -> None:
        """Aggregate this group over ranks and update at the end of a window."""
        (
            effective_diversity,
            valid_group,
            majority_ratio,
            vote_margin,
        ) = answer_distribution_metrics(normalized_answers)

        local_metrics = torch.tensor(
            [
                float(sum(confidences)),
                float(len(confidences)),
                effective_diversity,
                1.0,
                valid_group,
                majority_ratio,
                vote_margin,
            ],
            device=device,
            dtype=torch.float32,
        )

        if torch_dist.is_available() and torch_dist.is_initialized():
            torch_dist.all_reduce(local_metrics, op=torch_dist.ReduceOp.SUM)

        global_metrics = local_metrics.detach().cpu().tolist()
        self.metric_sums = [
            old + new for old, new in zip(self.metric_sums, global_metrics)
        ]
        self.calls_in_window += 1

        if self.calls_in_window < self.window_groups:
            return

        (
            confidence_sum,
            confidence_count,
            diversity_sum,
            group_count,
            valid_group_sum,
            majority_ratio_sum,
            vote_margin_sum,
        ) = self.metric_sums

        mean_confidence = confidence_sum / max(confidence_count, 1.0)
        mean_effective_diversity = diversity_sum / max(group_count, 1.0)
        valid_group_rate = valid_group_sum / max(group_count, 1.0)
        mean_majority_ratio = majority_ratio_sum / max(group_count, 1.0)
        mean_vote_margin = vote_margin_sum / max(group_count, 1.0)

        update = self.controller.update(
            mean_confidence=mean_confidence,
            effective_diversity=mean_effective_diversity,
            valid_group_rate=valid_group_rate,
        )
        self.window_index += 1

        print(
            "adaptive_temperature| "
            f"window: {self.window_index} | "
            f"phase: {update['phase']} | "
            f"temperature: {update['old_temperature']:.4f} | "
            f"next_temperature: {update['new_temperature']:.4f} | "
            f"mean_confidence: {mean_confidence:.4f} | "
            f"target_confidence: {self.controller.reference_confidence:.4f} | "
            f"effective_diversity: {mean_effective_diversity:.4f} | "
            f"target_effective_diversity: "
            f"{self.controller.reference_effective_diversity:.4f} | "
            f"valid_group_rate: {valid_group_rate:.4f} | "
            f"target_valid_group_rate: "
            f"{self.controller.reference_valid_group_rate:.4f} | "
            f"majority_ratio: {mean_majority_ratio:.4f} | "
            f"vote_margin: {mean_vote_margin:.4f} | "
            f"confidence_error: {update['confidence_error']:.4f} | "
            f"exploration_bonus: {update['exploration_bonus']:.4f}",
            flush=True,
        )

        self.calls_in_window = 0
        self.metric_sums = [0.0] * 7


def build_adaptive_sampler(
    runtime: DistributedTemperatureRuntime,
    majority_vote_module: ModuleType,
):
    """Create a sampler with the same output and logs as the old entry point."""

    @torch.no_grad()
    def sample_with_adaptive_temperature(
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

        if len(batch["problems"]) != 1:
            raise ValueError(
                "The adaptive GSM8K entry point currently requires "
                "batch_size_per_device=1."
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

        all_confidences = [
            confidence for _, confidence in generated_with_confidence
        ]
        # Preserve both confidence prefixes expected by the old analysis code.
        print("confidence list:", all_confidences)

        if not generated_with_confidence:
            return {
                "generated_ids": torch.tensor([]),
                "prompt_len": prompt_ids.shape[1],
                "rewards": None,
                "label_true": None,
            }

        all_generated_ids = torch.stack(
            [generated_id for generated_id, _ in generated_with_confidence]
        )
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
            confidences=all_confidences,
            normalized_answers=normalized_answers,
            device=device,
        )

        return {
            "generated_ids": all_generated_ids,
            "prompt_len": prompt_ids.shape[1],
            "rewards": rewards,
            # The patched loss uses the exact temperature that generated this
            # group, even if the controller changes before a later group.
            "rollout_temperature": rollout_temperature,
        }

    return sample_with_adaptive_temperature


def patch_loss_temperature(base_module: ModuleType) -> None:
    """Make policy loss use each group's actual rollout temperature."""
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
    parser.add_argument("--adaptive_temp_min", type=float, default=0.5)
    parser.add_argument("--adaptive_temp_max", type=float, default=1.4)
    parser.add_argument("--adaptive_temp_confidence_gain", type=float, default=0.5)
    parser.add_argument("--adaptive_temp_valid_gain", type=float, default=0.05)
    parser.add_argument("--adaptive_temp_deadband", type=float, default=0.005)
    parser.add_argument("--adaptive_temp_max_change", type=float, default=0.05)
    parser.add_argument(
        "--adaptive_temp_window",
        type=int,
        default=0,
        help="Sample calls per control window; 0 uses --grad_accum.",
    )
    return parser.parse_known_args()


def build_train_config(base_module: ModuleType, args: argparse.Namespace):
    """Mirror the current rank.py command-line-to-config mapping."""
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

    if any(argument == "--dynamic_sampling" for argument in base_argv):
        raise SystemExit(
            "This entry point intentionally disables Dynamic Sampling; "
            "remove --dynamic_sampling."
        )

    base_module = load_local_module("dttrl_gsm8k_rank_base", BASE_TRAIN_SCRIPT)
    majority_vote_module = load_local_module(
        "dttrl_gsm8k_uniform_majority",
        MAJORITY_VOTE_SCRIPT,
    )

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *base_argv]
        base_args = base_module.parse_args()
    finally:
        sys.argv = original_argv

    if base_args.dynamic_sampling:
        raise SystemExit("Dynamic Sampling must remain disabled for this experiment.")

    window_groups = controller_args.adaptive_temp_window or base_args.grad_accum
    controller = AdaptiveTemperatureController(
        temperature=base_args.temperature,
        min_temperature=controller_args.adaptive_temp_min,
        max_temperature=controller_args.adaptive_temp_max,
        confidence_gain=controller_args.adaptive_temp_confidence_gain,
        valid_group_gain=controller_args.adaptive_temp_valid_gain,
        confidence_deadband=controller_args.adaptive_temp_deadband,
        max_change=controller_args.adaptive_temp_max_change,
    )
    runtime = DistributedTemperatureRuntime(
        controller=controller,
        window_groups=window_groups,
    )

    base_module.sample_with_weighted_confidence = build_adaptive_sampler(
        runtime=runtime,
        majority_vote_module=majority_vote_module,
    )
    patch_loss_temperature(base_module)

    config = build_train_config(base_module, base_args)
    base_module.train(config)


if __name__ == "__main__":
    main()
