"""GSM8K dTTRL with uniform voting and confidence-only temperature control.

This additive v2 entry point reuses the already-tested sampling, uniform-vote,
loss-temperature, and training integration from
``train_gsm8k.rollout8.adaptive_temperature.py``.  It replaces only the
temperature controller and its metric runtime.  Existing training files are
not modified.

Diversity, majority ratio, vote margin, and valid-group rate are still logged
for analysis.  None of them changes temperature.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

import torch
import torch.distributed as torch_dist

from adaptive_temperature_confidence_only import (
    ConfidenceOnlyTemperatureController,
)


ROOT = Path(__file__).resolve().parent
LEGACY_ADAPTIVE_ENTRY = ROOT / "train_gsm8k.rollout8.adaptive_temperature.py"


def load_local_module(module_name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ConfidenceOnlyDistributedTemperatureRuntime:
    """Aggregate metrics across ranks and update one shared temperature."""

    def __init__(
        self,
        controller: ConfidenceOnlyTemperatureController,
        window_groups: int,
        answer_distribution_metrics,
    ) -> None:
        if window_groups <= 0:
            raise ValueError("window_groups must be greater than zero")

        self.controller = controller
        self.window_groups = int(window_groups)
        self.answer_distribution_metrics = answer_distribution_metrics
        self.window_index = 0
        self.calls_in_window = 0
        self.metric_sums = [0.0] * 7
        self.printed_configuration = False
        self.reference_effective_diversity = None
        self.reference_valid_group_rate = None

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
        print("Adaptive rollout temperature: confidence-only v2")
        print(f"Adaptive initial temperature: {self.controller.temperature}")
        print(
            "Adaptive temperature bounds: "
            f"[{self.controller.min_temperature}, "
            f"{self.controller.max_temperature}]"
        )
        print(f"Adaptive metric window: {self.window_groups} sampled groups per rank")
        print(f"Adaptive calibration windows: {self.controller.calibration_windows}")
        print(f"Adaptive confidence EMA decay: {self.controller.ema_decay}")
        print(f"Adaptive confidence gain: {self.controller.confidence_gain}")
        print(f"Adaptive confidence deadband: {self.controller.confidence_deadband}")
        print(f"Adaptive max change per window: {self.controller.max_change}")
        print("Adaptive diversity/valid-group temperature bonus: disabled")
        self.printed_configuration = True

    def observe(
        self,
        confidences: Sequence[float],
        normalized_answers: Sequence,
        device: torch.device,
    ) -> None:
        """Aggregate one sampled group and update at the window boundary."""
        (
            effective_diversity,
            valid_group,
            majority_ratio,
            vote_margin,
        ) = self.answer_distribution_metrics(normalized_answers)

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

        if self.reference_effective_diversity is None:
            self.reference_effective_diversity = mean_effective_diversity
            self.reference_valid_group_rate = valid_group_rate

        update = self.controller.update(mean_confidence=mean_confidence)
        self.window_index += 1

        print(
            "adaptive_temperature| "
            f"window: {self.window_index} | "
            f"phase: {update['phase']} | "
            f"temperature: {update['old_temperature']:.4f} | "
            f"next_temperature: {update['new_temperature']:.4f} | "
            f"mean_confidence: {mean_confidence:.4f} | "
            f"smoothed_confidence: {update['smoothed_confidence']:.4f} | "
            f"target_confidence: {update['reference_confidence']:.4f} | "
            f"raw_confidence_error: {update['raw_confidence_error']:.4f} | "
            f"confidence_error: {update['confidence_error']:.4f} | "
            f"effective_diversity: {mean_effective_diversity:.4f} | "
            f"target_effective_diversity: "
            f"{self.reference_effective_diversity:.4f} | "
            f"valid_group_rate: {valid_group_rate:.4f} | "
            f"target_valid_group_rate: "
            f"{self.reference_valid_group_rate:.4f} | "
            f"majority_ratio: {mean_majority_ratio:.4f} | "
            f"vote_margin: {mean_vote_margin:.4f} | "
            "exploration_bonus: 0.0000 | "
            f"calibration_progress: {update['calibration_progress']}/"
            f"{self.controller.calibration_windows}",
            flush=True,
        )

        self.calls_in_window = 0
        self.metric_sums = [0.0] * 7


def parse_controller_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adaptive_temp_min", type=float, default=0.5)
    parser.add_argument("--adaptive_temp_max", type=float, default=1.5)
    parser.add_argument("--adaptive_temp_confidence_gain", type=float, default=1.0)
    parser.add_argument("--adaptive_temp_deadband", type=float, default=0.002)
    parser.add_argument("--adaptive_temp_max_change", type=float, default=0.01)
    parser.add_argument("--adaptive_temp_ema_decay", type=float, default=0.8)
    parser.add_argument("--adaptive_temp_calibration_windows", type=int, default=3)
    parser.add_argument(
        "--adaptive_temp_window",
        type=int,
        default=0,
        help="Sample calls per control window; 0 uses --grad_accum.",
    )
    return parser.parse_known_args()


def main() -> None:
    controller_args, base_argv = parse_controller_args()

    if any(
        argument == "--dynamic_sampling"
        or argument.startswith("--dynamic_sampling=")
        for argument in base_argv
    ):
        raise SystemExit(
            "This entry point intentionally disables Dynamic Sampling; "
            "remove --dynamic_sampling."
        )

    legacy = load_local_module(
        "dttrl_gsm8k_adaptive_temperature_legacy",
        LEGACY_ADAPTIVE_ENTRY,
    )
    base_module = legacy.load_local_module(
        "dttrl_gsm8k_rank_base_confidence_only",
        legacy.BASE_TRAIN_SCRIPT,
    )
    majority_vote_module = legacy.load_local_module(
        "dttrl_gsm8k_uniform_majority_confidence_only",
        legacy.MAJORITY_VOTE_SCRIPT,
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
    controller = ConfidenceOnlyTemperatureController(
        temperature=base_args.temperature,
        min_temperature=controller_args.adaptive_temp_min,
        max_temperature=controller_args.adaptive_temp_max,
        confidence_gain=controller_args.adaptive_temp_confidence_gain,
        confidence_deadband=controller_args.adaptive_temp_deadband,
        max_change=controller_args.adaptive_temp_max_change,
        ema_decay=controller_args.adaptive_temp_ema_decay,
        calibration_windows=controller_args.adaptive_temp_calibration_windows,
    )
    runtime = ConfidenceOnlyDistributedTemperatureRuntime(
        controller=controller,
        window_groups=window_groups,
        answer_distribution_metrics=legacy.answer_distribution_metrics,
    )

    base_module.sample_with_weighted_confidence = legacy.build_adaptive_sampler(
        runtime=runtime,
        majority_vote_module=majority_vote_module,
    )
    legacy.patch_loss_temperature(base_module)

    config = legacy.build_train_config(base_module, base_args)
    base_module.train(config)


if __name__ == "__main__":
    main()
