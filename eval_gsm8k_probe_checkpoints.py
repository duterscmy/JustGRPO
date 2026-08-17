#!/usr/bin/env python3
"""Run the fixed GSM8K probe evaluator over checkpoint directories.

This launcher starts a fresh evaluation process for every checkpoint so GPU
memory is fully released between models.  Existing completed summaries are
skipped, while interrupted per-checkpoint evaluations resume from their
rank-local JSONL files.

Example:

    python eval_gsm8k_probe_checkpoints.py \
        --model_root checkpoints_gsm8k_block32 \
        --tokenizer_path /path/to/LLaDA-8B-Instruct \
        --output_root probe_results/block32_train \
        --block_sizes 1 32 --num_processes 4
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def checkpoint_step(path):
    match = re.search(r"(?:ckpt|checkpoint|step)[-_]?(\d+)$", path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(root, pattern, step_interval, start_step, end_step):
    checkpoints = []
    for path in root.glob(pattern):
        if not path.is_dir():
            continue
        step = checkpoint_step(path)
        if step is None:
            continue
        if step_interval > 0 and step % step_interval != 0:
            continue
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        checkpoints.append((step, path))
    return sorted(checkpoints, key=lambda item: item[0])


def result_directory(output_root, checkpoint, block_size, temperature, num_rollouts):
    temperature_text = f"{temperature:g}"
    return (
        output_root
        / checkpoint.name
        / f"block{block_size}_temp{temperature_text}_n{num_rollouts}"
    )


def build_command(args, eval_script, checkpoint, output_dir, block_size):
    evaluator_args = [
        str(eval_script),
        "--model_path",
        str(checkpoint),
        "--tokenizer_path",
        args.tokenizer_path,
        "--output_dir",
        str(output_dir),
        "--dataset_path",
        args.dataset_path,
        "--dataset_config",
        args.dataset_config,
        "--dataset_split",
        args.dataset_split,
        "--num_questions",
        str(args.num_questions),
        "--num_rollouts",
        str(args.num_rollouts),
        "--rollout_batch_size",
        str(args.rollout_batch_size),
        "--k_values",
        *[str(k) for k in args.k_values],
        "--majority_trials",
        str(args.majority_trials),
        "--temperature",
        str(args.temperature),
        "--steps",
        str(args.steps),
        "--gen_length",
        str(args.gen_length),
        "--block_size",
        str(block_size),
        "--mask_token_id",
        str(args.mask_token_id),
        "--seed",
        str(args.seed),
        "--probe_seed",
        str(args.probe_seed),
        "--metric_seed",
        str(args.metric_seed),
        "--dtype",
        args.dtype,
    ]
    if args.eos_token_id is not None:
        evaluator_args.extend(["--eos_token_id", str(args.eos_token_id)])
    if args.save_token_traces:
        evaluator_args.append("--save_token_traces")

    if args.num_processes == 1:
        return [args.python, *evaluator_args]

    return [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.num_processes),
        *evaluator_args,
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate every selected checkpoint on one fixed GSM8K probe."
    )
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint_glob", default="ckpt-*")
    parser.add_argument("--step_interval", type=int, default=5)
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--end_step", type=int, default=None)
    parser.add_argument(
        "--block_sizes",
        type=int,
        nargs="+",
        default=[32],
        help="Evaluate one or several decoders, e.g. --block_sizes 1 32.",
    )
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--eval_script",
        default=str(Path(__file__).with_name("eval_gsm8k_probe.py")),
    )
    parser.add_argument("--dataset_path", default="gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--num_rollouts", type=int, default=64)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--majority_trials", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--mask_token_id", type=int, default=126336)
    parser.add_argument("--eos_token_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--probe_seed", type=int, default=2026)
    parser.add_argument("--metric_seed", type=int, default=314159)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--save_token_traces", action="store_true")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to later checkpoints if one evaluation fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_root = Path(args.model_root).resolve()
    output_root = Path(args.output_root).resolve()
    eval_script = Path(args.eval_script).resolve()

    if not model_root.is_dir():
        raise FileNotFoundError(f"Model root not found: {model_root}")
    if not eval_script.is_file():
        raise FileNotFoundError(f"Probe evaluator not found: {eval_script}")
    if args.num_processes <= 0:
        raise ValueError("--num_processes must be positive.")
    if args.step_interval < 0:
        raise ValueError("--step_interval cannot be negative.")
    args.k_values = sorted(
        {k for k in args.k_values if 1 <= k <= args.num_rollouts}
    )
    if not args.k_values:
        raise ValueError("No --k_values remain after applying --num_rollouts.")

    checkpoints = discover_checkpoints(
        root=model_root,
        pattern=args.checkpoint_glob,
        step_interval=args.step_interval,
        start_step=args.start_step,
        end_step=args.end_step,
    )
    if not checkpoints:
        raise RuntimeError(
            f"No checkpoint matching {args.checkpoint_glob!r} under {model_root}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_root": str(model_root),
        "output_root": str(output_root),
        "checkpoint_glob": args.checkpoint_glob,
        "step_interval": args.step_interval,
        "checkpoints": [
            {"step": step, "path": str(path)} for step, path in checkpoints
        ],
        "block_sizes": sorted(set(args.block_sizes)),
        "num_questions": args.num_questions,
        "num_rollouts": args.num_rollouts,
        "temperature": args.temperature,
        "probe_seed": args.probe_seed,
    }
    manifest_path = output_root / "probe_sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    failures = []
    for step, checkpoint in checkpoints:
        for block_size in sorted(set(args.block_sizes)):
            output_dir = result_directory(
                output_root,
                checkpoint,
                block_size,
                args.temperature,
                args.num_rollouts,
            )
            summary_path = output_dir / "probe_summary.json"
            if summary_path.is_file():
                print(
                    f"[skip] step={step} block={block_size}: {summary_path}",
                    flush=True,
                )
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(
                args,
                eval_script,
                checkpoint,
                output_dir,
                block_size,
            )
            print(
                f"[run] step={step} block={block_size}\n"
                + " ".join(command),
                flush=True,
            )
            if args.dry_run:
                continue

            log_path = output_dir / "evaluation.log"
            environment = os.environ.copy()
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            with log_path.open("a", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    cwd=str(eval_script.parent),
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            if completed.returncode != 0:
                failure = {
                    "step": step,
                    "block_size": block_size,
                    "returncode": completed.returncode,
                    "log_path": str(log_path),
                }
                failures.append(failure)
                print(f"[failed] {failure}", flush=True)
                if not args.continue_on_error:
                    raise subprocess.CalledProcessError(
                        completed.returncode,
                        command,
                    )
            else:
                print(
                    f"[done] step={step} block={block_size}: {summary_path}",
                    flush=True,
                )

    if failures:
        failure_path = output_root / "probe_sweep_failures.json"
        with failure_path.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
            f.write("\n")
        raise SystemExit(f"{len(failures)} evaluations failed; see {failure_path}.")

    print(f"Probe sweep complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
