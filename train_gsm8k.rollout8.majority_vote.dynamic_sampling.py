"""Uniform-majority Dynamic Sampling baseline for an AR rollout.

This additive entry point keeps the existing Dynamic Sampling implementation
in ``train_gsm8k.rollout8.rank.py`` while replacing confidence-weighted voting
with the project's existing uniform-majority sampler.  Confidence is still
printed for analysis but never affects voting or group selection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys

import grpo


ROOT = Path(__file__).resolve().parent
BASE_TRAIN_SCRIPT = ROOT / "train_gsm8k.rollout8.rank.py"
MAJORITY_VOTE_SCRIPT = ROOT / "train_gsm8k.rollout8.majority_vote.py"


def load_majority_vote_module():
    spec = importlib.util.spec_from_file_location(
        "dttrl_gsm8k_uniform_majority_dynamic_sampling",
        MAJORITY_VOTE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {MAJORITY_VOTE_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def argument_value(name: str, default: str | None = None) -> str | None:
    for index, argument in enumerate(sys.argv[1:]):
        if argument == name:
            absolute_index = index + 1
            if absolute_index + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires a value")
            return sys.argv[absolute_index + 1]
        prefix = name + "="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return default


def main() -> None:
    if not BASE_TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Required file was not found: {BASE_TRAIN_SCRIPT}")
    if not MAJORITY_VOTE_SCRIPT.is_file():
        raise FileNotFoundError(f"Required file was not found: {MAJORITY_VOTE_SCRIPT}")

    if "--dynamic_sampling" not in sys.argv[1:]:
        raise SystemExit(
            "This baseline requires --dynamic_sampling."
        )

    block_size = int(argument_value("--block_size", "1"))
    if block_size != 1:
        raise SystemExit("This experiment requires --block_size 1 (AR rollout).")

    majority_vote_module = load_majority_vote_module()
    grpo.sample_with_weighted_confidence = (
        majority_vote_module.sample_with_logged_confidence_uniform_vote
    )

    print("Voting mode: uniform majority (confidence is logging-only)")
    print("Rollout mode: AR (block_size=1)")
    print("Adaptive rollout temperature: disabled")
    print("Dynamic sampling: enabled")
    runpy.run_path(str(BASE_TRAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
