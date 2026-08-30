#!/usr/bin/env python3
"""Forward-only logit diagnostics for every checkpoint in one GSM8K run.

The script never edits checkpoints or training files.  It evaluates a fixed
GSM8K probe with a fixed 50% corruption pattern and reports, for checkpoint 0
(the original base model) and every ``ckpt-*`` directory:

* full-vocabulary JS divergence from checkpoint 0;
* full-vocabulary JS divergence from the previous checkpoint;
* raw-logit entropy on all masked answer tokens; and
* ground-truth NLL and temperature-zero top-1 accuracy on all masked tokens.

JS is evaluated at a small, fixed set of masked positions per question.  NLL
and entropy use every masked position.  No sampling temperature is applied:
all probability metrics are computed from ``softmax(raw_logits)`` so runs
trained with different temperatures remain directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
CHECKPOINT_RE = re.compile(r"(?:ckpt|checkpoint|step)[-_]?(\d+)$")


@dataclass(frozen=True)
class ProbeExample:
    dataset_index: int
    input_ids: tuple[int, ...]
    target_positions: tuple[int, ...]
    target_ids: tuple[int, ...]
    js_positions: tuple[int, ...]


class RunningStats:
    """Streaming mean and standard error without retaining all values."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, values) -> None:
        import torch

        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_sq += float((values * values).sum().item())

    def update_zeros(self, count: int) -> None:
        self.count += int(count)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def sem(self) -> float:
        if self.count <= 1:
            return 0.0
        variance = max(
            0.0,
            (self.total_sq - self.total * self.total / self.count)
            / (self.count - 1),
        )
        return math.sqrt(variance / self.count)


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(run_dir: Path, checkpoint_glob: str) -> list[tuple[int, Path]]:
    by_step: dict[int, Path] = {}
    for path in run_dir.glob(checkpoint_glob):
        if not path.is_dir():
            continue
        step = checkpoint_step(path)
        if step is None or step == 0:
            continue
        if step in by_step:
            raise RuntimeError(
                f"Two checkpoint directories map to step {step}: "
                f"{by_step[step]} and {path}"
            )
        by_step[step] = path.resolve()
    return sorted(by_step.items())


def tokenize_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def render_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(rendered, list):
        if len(rendered) != 1:
            raise RuntimeError("Tokenizer unexpectedly returned several prompts.")
        rendered = rendered[0]
    return str(rendered)


def evenly_spaced_subset(sorted_values: Sequence[int], count: int) -> list[int]:
    if len(sorted_values) <= count:
        return list(sorted_values)
    indices = np.linspace(0, len(sorted_values) - 1, num=count)
    rounded = np.rint(indices).astype(np.int64)
    return [int(sorted_values[index]) for index in rounded]


def make_probe_example(
    tokenizer,
    example,
    dataset_index: int,
    mask_token_id: int,
    mask_ratio: float,
    mask_seed: int,
    js_positions_per_question: int,
    max_answer_tokens: int,
    max_sequence_length: int,
) -> ProbeExample | None:
    prompt_ids = tokenize_text(tokenizer, render_prompt(tokenizer, example["question"]))
    answer_ids = tokenize_text(tokenizer, example["answer"])
    answer_ids = answer_ids[:max_answer_tokens]

    available = max_sequence_length - len(prompt_ids)
    if available <= 0:
        return None
    answer_ids = answer_ids[:available]
    if not answer_ids:
        return None

    mask_count = max(1, int(round(len(answer_ids) * mask_ratio)))
    mask_count = min(mask_count, len(answer_ids))
    rng = np.random.default_rng(mask_seed + dataset_index * 1_000_003)
    masked_local = sorted(
        int(position)
        for position in rng.choice(len(answer_ids), size=mask_count, replace=False)
    )

    input_ids = list(prompt_ids) + list(answer_ids)
    target_positions = [len(prompt_ids) + position for position in masked_local]
    target_ids = [answer_ids[position] for position in masked_local]
    for position in target_positions:
        input_ids[position] = mask_token_id

    js_local = evenly_spaced_subset(masked_local, js_positions_per_question)
    js_positions = [len(prompt_ids) + position for position in js_local]
    return ProbeExample(
        dataset_index=int(dataset_index),
        input_ids=tuple(input_ids),
        target_positions=tuple(target_positions),
        target_ids=tuple(target_ids),
        js_positions=tuple(js_positions),
    )


def build_probe(tokenizer, dataset, args) -> tuple[list[ProbeExample], list[int]]:
    if args.num_questions > len(dataset):
        raise ValueError(
            f"Requested {args.num_questions} questions, dataset has {len(dataset)}."
        )

    rng = np.random.default_rng(args.probe_seed)
    probe: list[ProbeExample] = []
    dataset_indices: list[int] = []
    for dataset_index in rng.permutation(len(dataset)):
        item = make_probe_example(
            tokenizer=tokenizer,
            example=dataset[int(dataset_index)],
            dataset_index=int(dataset_index),
            mask_token_id=args.mask_token_id,
            mask_ratio=args.mask_ratio,
            mask_seed=args.mask_seed,
            js_positions_per_question=args.js_positions_per_question,
            max_answer_tokens=args.max_answer_tokens,
            max_sequence_length=args.max_sequence_length,
        )
        if item is None:
            continue
        probe.append(item)
        dataset_indices.append(int(dataset_index))
        if len(probe) == args.num_questions:
            break

    if len(probe) != args.num_questions:
        raise RuntimeError(
            f"Could construct only {len(probe)} of {args.num_questions} probe examples."
        )
    return probe, dataset_indices


def batches(values: Sequence[ProbeExample], batch_size: int) -> Iterable[Sequence[ProbeExample]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def collate_probe_batch(examples: Sequence[ProbeExample], pad_token_id: int, device):
    import torch

    maximum_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), maximum_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)

    target_batch: list[int] = []
    target_position: list[int] = []
    target_ids: list[int] = []
    js_batch: list[int] = []
    js_position: list[int] = []

    for batch_index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[batch_index, :length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[batch_index, :length] = 1
        target_batch.extend([batch_index] * len(example.target_positions))
        target_position.extend(example.target_positions)
        target_ids.extend(example.target_ids)
        js_batch.extend([batch_index] * len(example.js_positions))
        js_position.extend(example.js_positions)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_batch": torch.tensor(target_batch, dtype=torch.long, device=device),
        "target_position": torch.tensor(
            target_position, dtype=torch.long, device=device
        ),
        "target_ids": torch.tensor(target_ids, dtype=torch.long, device=device),
        "js_batch": torch.tensor(js_batch, dtype=torch.long, device=device),
        "js_position": torch.tensor(js_position, dtype=torch.long, device=device),
    }


def load_cached_logits(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def js_divergence_from_logits(current, reference, device, chunk_size: int):
    """Return exact full-vocabulary JS values, one per selected position."""
    import torch
    import torch.nn.functional as functional

    if current.shape != reference.shape:
        raise RuntimeError(
            f"Cached logit shape mismatch: {tuple(current.shape)} vs "
            f"{tuple(reference.shape)}"
        )

    values = []
    log_two = math.log(2.0)
    for start in range(0, current.shape[0], chunk_size):
        end = min(start + chunk_size, current.shape[0])
        current_chunk = current[start:end].to(device=device, dtype=torch.float32)
        reference_chunk = reference[start:end].to(
            device=device, dtype=torch.float32
        )
        log_current = functional.log_softmax(current_chunk, dim=-1)
        log_reference = functional.log_softmax(reference_chunk, dim=-1)
        log_mixture = torch.logaddexp(log_current, log_reference) - log_two
        current_kl = (
            log_current.exp() * (log_current - log_mixture)
        ).sum(dim=-1)
        reference_kl = (
            log_reference.exp() * (log_reference - log_mixture)
        ).sum(dim=-1)
        values.append((0.5 * (current_kl + reference_kl)).cpu())
        del current_chunk, reference_chunk
        del log_current, log_reference, log_mixture
    return torch.cat(values, dim=0)


def load_model(model_path: Path, dtype, device):
    from transformers import AutoModel

    print(f"Loading model: {model_path}", flush=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    return model.eval().requires_grad_(False).to(device)


def evaluate_checkpoint(
    model_path: Path,
    probe: Sequence[ProbeExample],
    current_cache: Path,
    initial_cache: Path | None,
    previous_cache: Path | None,
    pad_token_id: int,
    args,
    dtype,
    device,
):
    import torch
    import torch.nn.functional as functional

    current_cache.mkdir(parents=True, exist_ok=False)
    nll_stats = RunningStats()
    entropy_stats = RunningStats()
    accuracy_stats = RunningStats()
    js_initial_stats = RunningStats()
    js_previous_stats = RunningStats()

    model = load_model(model_path, dtype=dtype, device=device)
    with torch.inference_mode():
        for batch_index, examples in enumerate(batches(probe, args.batch_size)):
            tensors = collate_probe_batch(examples, pad_token_id, device)
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                ).logits

            target_logits = logits[
                tensors["target_batch"], tensors["target_position"]
            ].float()
            target_log_probs = functional.log_softmax(target_logits, dim=-1)
            nll = -target_log_probs.gather(
                dim=-1,
                index=tensors["target_ids"].unsqueeze(-1),
            ).squeeze(-1)
            entropy = -(target_log_probs.exp() * target_log_probs).sum(dim=-1)
            top1_accuracy = target_logits.argmax(dim=-1).eq(
                tensors["target_ids"]
            ).float()
            nll_stats.update(nll)
            entropy_stats.update(entropy)
            accuracy_stats.update(top1_accuracy)

            selected_logits = logits[
                tensors["js_batch"], tensors["js_position"]
            ].detach().to(dtype=torch.bfloat16).cpu().contiguous()
            cache_path = current_cache / f"batch-{batch_index:05d}.pt"
            torch.save(selected_logits, cache_path)

            if initial_cache is None:
                count = int(selected_logits.shape[0])
                js_initial_stats.update_zeros(count)
                js_previous_stats.update_zeros(count)
            else:
                initial_logits = load_cached_logits(
                    initial_cache / cache_path.name
                )
                js_initial = js_divergence_from_logits(
                    selected_logits,
                    initial_logits,
                    device=device,
                    chunk_size=args.js_chunk_size,
                )
                js_initial_stats.update(js_initial)

                if previous_cache is not None and previous_cache != initial_cache:
                    previous_logits = load_cached_logits(
                        previous_cache / cache_path.name
                    )
                    js_previous = js_divergence_from_logits(
                        selected_logits,
                        previous_logits,
                        device=device,
                        chunk_size=args.js_chunk_size,
                    )
                else:
                    js_previous = js_initial
                js_previous_stats.update(js_previous)

            del logits, target_logits, target_log_probs, nll, entropy
            del top1_accuracy
            del selected_logits, tensors
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "masked_token_count": nll_stats.count,
        "js_position_count": js_initial_stats.count,
        "js_to_initial": js_initial_stats.mean,
        "js_to_initial_sem": js_initial_stats.sem,
        "js_to_previous": js_previous_stats.mean,
        "js_to_previous_sem": js_previous_stats.sem,
        "raw_entropy": entropy_stats.mean,
        "raw_entropy_sem": entropy_stats.sem,
        "ground_truth_nll": nll_stats.mean,
        "ground_truth_nll_sem": nll_stats.sem,
        "masked_token_top1_accuracy": accuracy_stats.mean,
        "masked_token_top1_accuracy_sem": accuracy_stats.sem,
    }


CSV_FIELDS = (
    "step",
    "checkpoint_path",
    "masked_token_count",
    "js_position_count",
    "js_to_initial",
    "js_to_initial_sem",
    "js_to_previous",
    "js_to_previous_sem",
    "raw_entropy",
    "raw_entropy_sem",
    "ground_truth_nll",
    "ground_truth_nll_sem",
    "masked_token_top1_accuracy",
    "masked_token_top1_accuracy_sem",
)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_FIELDS})
    os.replace(temporary, path)


def plot_results(output_path: Path, rows: Sequence[dict], experiment_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
    panels = (
        ("js_to_initial", "js_to_initial_sem", "JS divergence to checkpoint 0"),
        ("js_to_previous", "js_to_previous_sem", "JS divergence to previous checkpoint"),
        ("raw_entropy", "raw_entropy_sem", "Raw-logit entropy"),
        ("ground_truth_nll", "ground_truth_nll_sem", "Ground-truth token NLL"),
        (
            "masked_token_top1_accuracy",
            "masked_token_top1_accuracy_sem",
            "T=0 masked-token top-1 accuracy",
        ),
    )

    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for axis, (value_key, sem_key, title) in zip(axes.flat, panels):
        values = np.asarray([row[value_key] for row in rows], dtype=np.float64)
        sem = np.asarray([row[sem_key] for row in rows], dtype=np.float64)
        axis.plot(steps, values, marker="o", linewidth=2.0, color="#1f77b4")
        axis.fill_between(
            steps,
            values - 1.96 * sem,
            values + 1.96 * sem,
            color="#1f77b4",
            alpha=0.15,
            linewidth=0,
        )
        axis.set_title(title)
        axis.set_xlabel("Checkpoint step")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.set_xticks(steps)

    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.0,
        0.9,
        "All metrics use the same fixed GSM8K questions and masks.\n"
        "JS / entropy / NLL: softmax(raw logits), without temperature.\n"
        "Top-1 accuracy: argmax(raw logits), equivalent to T=0.",
        va="top",
        fontsize=11,
    )

    figure.suptitle(
        f"GSM8K checkpoint logit diagnostics\n{experiment_name}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, rows: Sequence[dict], experiment_name: str) -> None:
    write_csv(output_dir / "checkpoint_logit_metrics.csv", rows)
    atomic_write_json(output_dir / "checkpoint_logit_metrics.json", list(rows))
    plot_results(
        output_dir / "checkpoint_logit_diagnostics.png",
        rows,
        experiment_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot forward-only GSM8K logit diagnostics over checkpoints."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint_glob", default="ckpt-*")
    parser.add_argument("--dataset_path", default="gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--num_questions", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--probe_seed", type=int, default=2026)
    parser.add_argument("--mask_seed", type=int, default=314159)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--js_positions_per_question", type=int, default=8)
    parser.add_argument("--js_chunk_size", type=int, default=4)
    parser.add_argument("--max_answer_tokens", type=int, default=256)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--mask_token_id", type=int, default=126336)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if not 0.0 < args.mask_ratio <= 1.0:
        raise ValueError("--mask_ratio must be in (0, 1].")
    if args.js_positions_per_question <= 0:
        raise ValueError("--js_positions_per_question must be positive.")
    if args.js_chunk_size <= 0:
        raise ValueError("--js_chunk_size must be positive.")
    if args.max_answer_tokens <= 0 or args.max_sequence_length <= 0:
        raise ValueError("Maximum token lengths must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LLaDA checkpoint diagnostic.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    run_dir = Path(args.run_dir).expanduser().resolve()
    base_model_path = Path(args.base_model_path).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {run_dir}")
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_path}")

    checkpoints = discover_checkpoints(run_dir, args.checkpoint_glob)
    if not checkpoints:
        raise RuntimeError(f"No {args.checkpoint_glob!r} directories under {run_dir}.")
    checkpoint_sequence = [(0, base_model_path), *checkpoints]

    if args.output_dir is None:
        output_dir = run_dir.parent / f"{run_dir.name}_logit_diagnostics"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )
    tokenizer.pad_token_id = args.mask_token_id
    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = args.mask_token_id

    dataset = load_dataset(
        args.dataset_path,
        args.dataset_config,
        split=args.dataset_split,
    )
    probe, dataset_indices = build_probe(tokenizer, dataset, args)

    config = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "base_model_path": str(base_model_path),
        "checkpoint_sequence": [
            {"step": step, "path": str(path)}
            for step, path in checkpoint_sequence
        ],
        "dataset_path": args.dataset_path,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_indices": dataset_indices,
        "num_questions": args.num_questions,
        "probe_seed": args.probe_seed,
        "mask_seed": args.mask_seed,
        "mask_ratio": args.mask_ratio,
        "js_positions_per_question": args.js_positions_per_question,
        "max_answer_tokens": args.max_answer_tokens,
        "max_sequence_length": args.max_sequence_length,
        "mask_token_id": args.mask_token_id,
        "probability_definition": "softmax(raw_logits); no sampling temperature",
    }
    atomic_write_json(output_dir / "diagnostic_config.json", config)

    print(
        f"Fixed probe: {len(probe)} GSM8K questions; "
        f"steps={[step for step, _ in checkpoint_sequence]}",
        flush=True,
    )
    rows: list[dict] = []
    initial_cache: Path | None = None
    previous_cache: Path | None = None

    with tempfile.TemporaryDirectory(
        prefix="gsm8k-logit-cache-", dir=output_dir
    ) as temporary_root_text:
        temporary_root = Path(temporary_root_text)
        for step, model_path in checkpoint_sequence:
            current_cache = temporary_root / f"step-{step:06d}"
            metrics = evaluate_checkpoint(
                model_path=model_path,
                probe=probe,
                current_cache=current_cache,
                initial_cache=initial_cache,
                previous_cache=previous_cache,
                pad_token_id=int(pad_token_id),
                args=args,
                dtype=dtype,
                device=device,
            )
            row = {
                "step": int(step),
                "checkpoint_path": str(model_path),
                **metrics,
            }
            rows.append(row)
            write_outputs(output_dir, rows, run_dir.name)
            print(json.dumps(row, indent=2), flush=True)

            if initial_cache is None:
                initial_cache = current_cache
            old_previous = previous_cache
            previous_cache = current_cache
            if (
                old_previous is not None
                and old_previous != initial_cache
                and old_previous.exists()
            ):
                shutil.rmtree(old_previous)

    print(f"Saved diagnostics to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
