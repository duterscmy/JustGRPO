#!/usr/bin/env python3
"""Evaluate one LLaDA checkpoint on a fixed GSM8K probe set.

The script is intentionally generation-first: it saves every rollout to JSONL
and also writes a compact summary JSON.  The saved records are sufficient for
later Pass@k, Maj@k, diversity, calibration, confidence, and length plots.

Example (4 GPUs):

    torchrun --standalone --nproc_per_node=4 eval_gsm8k_probe.py \
        --model_path checkpoints/.../ckpt-000005 \
        --tokenizer_path /path/to/LLaDA-8B-Instruct \
        --output_dir probe_results/ckpt-000005/block32 \
        --block_size 32 --num_questions 100 --num_rollouts 64 \
        --rollout_batch_size 8 --temperature 1.0
"""

import argparse
import itertools
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp_path, path)


def write_jsonl_atomic(path, rows):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp_path, path)


def read_jsonl(path, repair=False):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows

    found_incomplete_line = False
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A preempted job may leave one incomplete final line.  Keep
                # all complete questions and regenerate only that last one.
                print(
                    f"Warning: ignoring incomplete JSONL line {line_number} "
                    f"in {path}",
                    flush=True,
                )
                found_incomplete_line = True
    if repair and found_incomplete_line:
        write_jsonl_atomic(path, rows)
    return rows


def finite_values(values):
    return [float(x) for x in values if x is not None and math.isfinite(float(x))]


def mean_or_none(values):
    values = finite_values(values)
    return float(np.mean(values)) if values else None


def std_or_none(values):
    values = finite_values(values)
    return float(np.std(values)) if values else None


def quantile_or_none(values, q):
    values = finite_values(values)
    return float(np.quantile(values, q)) if values else None


def pass_at_k(num_samples, num_correct, k):
    """Unbiased Pass@k estimate: 1 - C(n-c, k) / C(n, k)."""
    if not 1 <= k <= num_samples:
        raise ValueError(f"k={k} must be in [1, {num_samples}].")
    if num_samples - num_correct < k:
        return 1.0

    failure_probability = 1.0
    for i in range(k):
        failure_probability *= (num_samples - num_correct - i) / (num_samples - i)
    return float(1.0 - failure_probability)


def safe_math_equal(left, right, math_equal_fn):
    try:
        return bool(math_equal_fn(left, right))
    except Exception:
        return False


def build_answer_classes(extracted_answers, ground_truth, math_equal_fn):
    """Group extracted answers into the same math_equal equivalence classes."""
    representatives = []
    class_ids = []

    for answer in extracted_answers:
        class_id = None
        for candidate_id, representative in enumerate(representatives):
            if safe_math_equal(answer, representative, math_equal_fn):
                class_id = candidate_id
                break
        if class_id is None:
            class_id = len(representatives)
            representatives.append(answer)
        class_ids.append(class_id)

    class_is_correct = [
        safe_math_equal(representative, ground_truth, math_equal_fn)
        for representative in representatives
    ]
    return representatives, np.asarray(class_ids, dtype=np.int64), class_is_correct


def vote_result(class_ids, class_is_correct, weights=None):
    num_classes = len(class_is_correct)
    if weights is None:
        scores = np.bincount(class_ids, minlength=num_classes).astype(np.float64)
    else:
        scores = np.bincount(
            class_ids,
            weights=np.asarray(weights, dtype=np.float64),
            minlength=num_classes,
        )

    maximum = float(scores.max())
    winners = np.flatnonzero(np.isclose(scores, maximum, rtol=1e-10, atol=1e-12))
    deterministic_winner = int(winners[0])
    expected_tie_accuracy = float(
        np.mean([float(class_is_correct[int(index)]) for index in winners])
    )
    score_sum = float(scores.sum())

    return {
        "winner": deterministic_winner,
        "is_correct": int(class_is_correct[deterministic_winner]),
        "expected_tie_accuracy": expected_tie_accuracy,
        "tie_size": int(len(winners)),
        "winner_share": maximum / score_sum if score_sum > 0 else 0.0,
        "scores": scores,
    }


def estimate_majority_at_k(
    class_ids,
    class_is_correct,
    weights,
    k_values,
    trials,
    seed,
):
    """Estimate Maj@k over subsets, using fractional credit for vote ties."""
    num_samples = len(class_ids)
    rng = np.random.default_rng(seed)
    uniform_results = {}
    weighted_results = {}

    for k in k_values:
        subset_count = math.comb(num_samples, k)
        if subset_count <= trials:
            subsets = itertools.combinations(range(num_samples), k)
        else:
            subsets = (
                rng.choice(num_samples, size=k, replace=False)
                for _ in range(trials)
            )

        uniform_total = 0.0
        weighted_total = 0.0
        evaluated = 0
        for subset in subsets:
            subset = np.asarray(subset, dtype=np.int64)
            subset_classes = class_ids[subset]
            uniform_total += vote_result(
                subset_classes,
                class_is_correct,
            )["expected_tie_accuracy"]
            weighted_total += vote_result(
                subset_classes,
                class_is_correct,
                weights=np.asarray(weights, dtype=np.float64)[subset],
            )["expected_tie_accuracy"]
            evaluated += 1

        uniform_results[str(k)] = uniform_total / evaluated
        weighted_results[str(k)] = weighted_total / evaluated

    return uniform_results, weighted_results


def analyze_question(
    extracted_answers,
    rollout_correctness,
    full_confidences,
    pre_eos_confidences,
    ground_truth,
    k_values,
    majority_trials,
    metric_seed,
    math_equal_fn,
):
    representatives, class_ids, class_is_correct = build_answer_classes(
        extracted_answers,
        ground_truth,
        math_equal_fn,
    )
    num_samples = len(class_ids)
    counts = np.bincount(class_ids, minlength=len(representatives))
    probabilities = counts.astype(np.float64) / num_samples
    nonzero_probabilities = probabilities[probabilities > 0]
    answer_entropy = float(
        -(nonzero_probabilities * np.log(nonzero_probabilities)).sum()
    )

    sorted_counts = np.sort(counts)[::-1]
    top1_count = int(sorted_counts[0])
    top2_count = int(sorted_counts[1]) if len(sorted_counts) > 1 else 0

    majority = vote_result(class_ids, class_is_correct)
    weighted_majority = vote_result(
        class_ids,
        class_is_correct,
        weights=full_confidences,
    )
    weighted_majority_pre_eos = vote_result(
        class_ids,
        class_is_correct,
        weights=pre_eos_confidences,
    )

    maj_at_k, weighted_maj_at_k = estimate_majority_at_k(
        class_ids=class_ids,
        class_is_correct=class_is_correct,
        weights=full_confidences,
        k_values=k_values,
        trials=majority_trials,
        seed=metric_seed,
    )

    num_correct = int(sum(rollout_correctness))
    pass_metrics = {
        str(k): pass_at_k(num_samples, num_correct, k)
        for k in k_values
    }

    normalized_answers = [representatives[int(index)] for index in class_ids]
    invalid_count = sum(answer == "" for answer in extracted_answers)

    return {
        "num_rollouts": num_samples,
        "num_correct_rollouts": num_correct,
        "rollout_accuracy": num_correct / num_samples,
        "pass_at_k": pass_metrics,
        "maj_at_k": maj_at_k,
        "weighted_maj_at_k": weighted_maj_at_k,
        "majority_answer": representatives[majority["winner"]],
        "majority_is_correct": majority["is_correct"],
        "majority_expected_tie_accuracy": majority["expected_tie_accuracy"],
        "majority_tie_size": majority["tie_size"],
        "majority_ratio": top1_count / num_samples,
        "second_answer_ratio": top2_count / num_samples,
        "majority_margin": (top1_count - top2_count) / num_samples,
        "weighted_majority_answer": representatives[weighted_majority["winner"]],
        "weighted_majority_is_correct": weighted_majority["is_correct"],
        "weighted_majority_expected_tie_accuracy": weighted_majority[
            "expected_tie_accuracy"
        ],
        "weighted_majority_share": weighted_majority["winner_share"],
        "weighted_pre_eos_majority_answer": representatives[
            weighted_majority_pre_eos["winner"]
        ],
        "weighted_pre_eos_majority_is_correct": weighted_majority_pre_eos[
            "is_correct"
        ],
        "distinct_answer_count": len(representatives),
        "distinct_answer_ratio": len(representatives) / num_samples,
        "answer_entropy": answer_entropy,
        "normalized_answer_entropy": (
            answer_entropy / math.log(num_samples) if num_samples > 1 else 0.0
        ),
        "all_same": int(len(representatives) == 1),
        "invalid_answer_count": invalid_count,
        "invalid_answer_ratio": invalid_count / num_samples,
        "normalized_answers": normalized_answers,
        "answer_counts": {
            representatives[index]: int(counts[index])
            for index in range(len(representatives))
        },
    }


def get_num_transfer_tokens(mask_index, steps):
    import torch

    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    result = torch.zeros(
        mask_num.size(0),
        steps,
        device=mask_index.device,
        dtype=torch.int64,
    ) + base
    for row in range(mask_num.size(0)):
        result[row, : int(remainder[row].item())] += 1
    return result


def add_gumbel_noise(logits, temperature):
    """Match the repository's LLaDA sampling implementation."""
    import torch

    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def generate_with_probe_traces(
    model,
    prompt,
    steps,
    gen_length,
    block_size,
    temperature,
    mask_token_id,
):
    """Generate while recording statistics at each token commitment."""
    import torch
    import torch.nn.functional as F

    if gen_length % block_size != 0:
        raise ValueError("--gen_length must be divisible by --block_size.")
    num_blocks = gen_length // block_size
    if steps % num_blocks != 0:
        raise ValueError("--steps must be divisible by gen_length / block_size.")
    steps_per_block = steps // num_blocks

    batch_size = prompt.shape[0]
    prompt_length = prompt.shape[1]
    device = prompt.device
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt

    trace_shape = (batch_size, gen_length)
    commit_confidence = torch.full(trace_shape, torch.nan, device=device)
    commit_entropy = torch.full(trace_shape, torch.nan, device=device)
    commit_margin = torch.full(trace_shape, torch.nan, device=device)
    commit_round = torch.full(
        trace_shape,
        -1,
        dtype=torch.int16,
        device=device,
    )

    with torch.inference_mode():
        for block_index in range(num_blocks):
            block_start = prompt_length + block_index * block_size
            block_end = block_start + block_size
            block_mask = x[:, block_start:block_end].eq(mask_token_id)
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask,
                steps_per_block,
            )

            for denoise_round in range(steps_per_block):
                mask_index = x.eq(mask_token_id)
                with torch.autocast(
                    device_type="cuda",
                    enabled=device.type == "cuda",
                    dtype=torch.bfloat16,
                ):
                    logits = model(x).logits

                noisy_logits = add_gumbel_noise(logits, temperature)
                sampled_tokens = torch.argmax(noisy_logits, dim=-1)
                del noisy_logits

                transfer_index = torch.zeros_like(x, dtype=torch.bool)

                for batch_index in range(batch_size):
                    candidate_indices = torch.where(
                        mask_index[batch_index, block_start:block_end]
                    )[0] + block_start
                    k = min(
                        int(num_transfer_tokens[batch_index, denoise_round].item()),
                        int(candidate_indices.numel()),
                    )
                    if k == 0:
                        continue

                    # Only materialize probabilities for the current block.
                    # This is equivalent to the training generator's full
                    # softmax for ranking, but uses much less temporary memory.
                    candidate_logits = logits[
                        batch_index,
                        candidate_indices,
                        :,
                    ].float()
                    candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
                    candidate_token_ids = sampled_tokens[
                        batch_index,
                        candidate_indices,
                    ]
                    sampled_log_probs = candidate_log_probs.gather(
                        -1,
                        candidate_token_ids.unsqueeze(-1),
                    ).squeeze(-1)
                    sampled_probabilities = sampled_log_probs.exp()

                    selected_local = torch.topk(
                        sampled_probabilities,
                        k=k,
                    ).indices
                    selected_absolute = candidate_indices[selected_local]
                    transfer_index[batch_index, selected_absolute] = True

                    selected_log_probs = candidate_log_probs[selected_local]
                    selected_probabilities = selected_log_probs.exp()
                    entropy = -(
                        selected_probabilities * selected_log_probs
                    ).sum(dim=-1)
                    top2 = torch.topk(
                        selected_probabilities,
                        k=2,
                        dim=-1,
                    ).values
                    margin = top2[:, 0] - top2[:, 1]

                    relative_positions = selected_absolute - prompt_length
                    commit_confidence[
                        batch_index,
                        relative_positions,
                    ] = sampled_probabilities[selected_local]
                    commit_entropy[
                        batch_index,
                        relative_positions,
                    ] = entropy
                    commit_margin[
                        batch_index,
                        relative_positions,
                    ] = margin
                    commit_round[
                        batch_index,
                        relative_positions,
                    ] = denoise_round

                    del candidate_logits
                    del candidate_log_probs
                    del selected_log_probs
                    del selected_probabilities

                x[transfer_index] = sampled_tokens[transfer_index]
                del logits
                del sampled_tokens

    if torch.isnan(commit_confidence).any():
        missing = int(torch.isnan(commit_confidence).sum().item())
        raise RuntimeError(f"Generation ended with {missing} uncommitted token traces.")

    return x, {
        "commit_confidence": commit_confidence,
        "commit_entropy": commit_entropy,
        "commit_margin": commit_margin,
        "commit_round": commit_round,
    }


def first_eos_lengths(completion_ids, eos_token_id):
    import torch

    eos_mask = completion_ids.eq(eos_token_id)
    has_eos = eos_mask.any(dim=1)
    first_eos = eos_mask.to(torch.int64).argmax(dim=1) + 1
    full_length = torch.full_like(first_eos, completion_ids.shape[1])
    lengths = torch.where(has_eos, first_eos, full_length)
    return lengths, has_eos


def tensor_mean(values):
    return float(values.float().mean().item()) if values.numel() else None


def infer_checkpoint_step(model_path):
    match = re.search(r"(?:ckpt|checkpoint|step)[-_]?(\d+)$", Path(model_path).name)
    return int(match.group(1)) if match else None


def evaluate_one_question(
    model,
    tokenizer,
    question,
    ground_truth_cot,
    dataset_index,
    probe_order,
    args,
    device,
    eos_token_id,
    extract_answer_fn,
    extract_ground_truth_fn,
    math_equal_fn,
):
    import torch

    prompt_text = tokenizer.apply_chat_template(
        [[{"role": "user", "content": question}]],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
    )["input_ids"].to(device)

    rollout_rows = []
    generated_so_far = 0
    while generated_so_far < args.num_rollouts:
        current_batch_size = min(
            args.rollout_batch_size,
            args.num_rollouts - generated_so_far,
        )
        generation_seed = (
            args.seed
            + dataset_index * 1_000_003
            + generated_so_far * 10_007
        )
        torch.manual_seed(generation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(generation_seed)

        repeated_prompt = prompt_ids.repeat(current_batch_size, 1)
        generated_ids, traces = generate_with_probe_traces(
            model=model,
            prompt=repeated_prompt,
            steps=args.steps,
            gen_length=args.gen_length,
            block_size=args.block_size,
            temperature=args.temperature,
            mask_token_id=args.mask_token_id,
        )
        completion_ids = generated_ids[
            :,
            prompt_ids.shape[1] : prompt_ids.shape[1] + args.gen_length,
        ]
        lengths, has_eos = first_eos_lengths(completion_ids, eos_token_id)

        for batch_index in range(current_batch_size):
            length = int(lengths[batch_index].item())
            effective_ids = completion_ids[batch_index, :length]
            response = tokenizer.decode(
                effective_ids,
                skip_special_tokens=True,
            )
            confidence = traces["commit_confidence"][batch_index]
            entropy = traces["commit_entropy"][batch_index]
            margin = traces["commit_margin"][batch_index]

            rollout = {
                "rollout_index": generated_so_far + batch_index,
                "response": response,
                "generation_length": length,
                "eos_found": bool(has_eos[batch_index].item()),
                "mean_commit_confidence": tensor_mean(confidence),
                "pre_eos_mean_commit_confidence": tensor_mean(confidence[:length]),
                "mean_commit_entropy": tensor_mean(entropy),
                "pre_eos_mean_commit_entropy": tensor_mean(entropy[:length]),
                "mean_top1_top2_margin": tensor_mean(margin),
                "pre_eos_mean_top1_top2_margin": tensor_mean(margin[:length]),
            }
            if args.save_token_traces:
                rollout["token_traces"] = {
                    "commit_confidence": confidence.float().cpu().tolist(),
                    "commit_entropy": entropy.float().cpu().tolist(),
                    "top1_top2_margin": margin.float().cpu().tolist(),
                    "commit_round": traces["commit_round"][
                        batch_index
                    ].cpu().tolist(),
                }
            rollout_rows.append(rollout)

        generated_so_far += current_batch_size
        del generated_ids
        del completion_ids
        del traces
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ground_truth = extract_ground_truth_fn(ground_truth_cot)
    extracted_answers = [
        extract_answer_fn(rollout["response"])
        for rollout in rollout_rows
    ]
    correctness = [
        int(safe_math_equal(answer, ground_truth, math_equal_fn))
        for answer in extracted_answers
    ]

    for rollout, extracted_answer, is_correct in zip(
        rollout_rows,
        extracted_answers,
        correctness,
    ):
        rollout["extracted_answer"] = extracted_answer
        rollout["is_correct"] = is_correct

    question_metrics = analyze_question(
        extracted_answers=extracted_answers,
        rollout_correctness=correctness,
        full_confidences=[
            rollout["mean_commit_confidence"] for rollout in rollout_rows
        ],
        pre_eos_confidences=[
            rollout["pre_eos_mean_commit_confidence"] for rollout in rollout_rows
        ],
        ground_truth=ground_truth,
        k_values=args.k_values,
        majority_trials=args.majority_trials,
        metric_seed=args.metric_seed + dataset_index,
        math_equal_fn=math_equal_fn,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_index": dataset_index,
        "probe_order": probe_order,
        "question": question,
        "ground_truth_cot": ground_truth_cot,
        "ground_truth_answer": ground_truth,
        "prompt_length": int(prompt_ids.shape[1]),
        "metrics": question_metrics,
        "rollouts": rollout_rows,
    }


def summarize_records(records, args):
    rollout_rows = [
        rollout
        for record in records
        for rollout in record["rollouts"]
    ]
    question_metrics = [record["metrics"] for record in records]

    def question_mean(key):
        return mean_or_none(metric[key] for metric in question_metrics)

    def rollout_values(key):
        return [rollout[key] for rollout in rollout_rows]

    pass_summary = {
        str(k): mean_or_none(
            metric["pass_at_k"][str(k)] for metric in question_metrics
        )
        for k in args.k_values
    }
    maj_summary = {
        str(k): mean_or_none(
            metric["maj_at_k"][str(k)] for metric in question_metrics
        )
        for k in args.k_values
    }
    weighted_maj_summary = {
        str(k): mean_or_none(
            metric["weighted_maj_at_k"][str(k)]
            for metric in question_metrics
        )
        for k in args.k_values
    }

    correct_confidences = [
        rollout["pre_eos_mean_commit_confidence"]
        for rollout in rollout_rows
        if rollout["is_correct"]
    ]
    incorrect_confidences = [
        rollout["pre_eos_mean_commit_confidence"]
        for rollout in rollout_rows
        if not rollout["is_correct"]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_step": infer_checkpoint_step(args.model_path),
        "dataset": {
            "path": args.dataset_path,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "num_questions": len(records),
            "probe_seed": args.probe_seed,
            "dataset_indices": [record["dataset_index"] for record in records],
        },
        "generation": {
            "num_rollouts_per_question": args.num_rollouts,
            "temperature": args.temperature,
            "steps": args.steps,
            "gen_length": args.gen_length,
            "block_size": args.block_size,
            "rollout_batch_size_per_process": args.rollout_batch_size,
            "eos_token_id": args.eos_token_id,
            "mask_token_id": args.mask_token_id,
        },
        "metric_definition": {
            "pass_at_k": "Unbiased 1-C(n-c,k)/C(n,k), averaged over questions.",
            "maj_at_k": (
                "Expected majority-vote correctness over exact or sampled "
                "k-subsets; vote ties receive fractional credit."
            ),
            "pseudo_label_accuracy": (
                "Accuracy of confidence-weighted majority over all rollouts, "
                "matching the training reward's voting rule."
            ),
            "generation_length": "Completion tokens through the first EOS, inclusive.",
            "pre_eos_confidence": "Mean commitment confidence through first EOS, inclusive.",
        },
        "accuracy": {
            "rollout_accuracy": question_mean("rollout_accuracy"),
            "majority_vote_accuracy": question_mean("majority_is_correct"),
            "majority_vote_expected_tie_accuracy": question_mean(
                "majority_expected_tie_accuracy"
            ),
            "pseudo_label_accuracy": question_mean(
                "weighted_majority_is_correct"
            ),
            "pseudo_label_accuracy_pre_eos_weighted": question_mean(
                "weighted_pre_eos_majority_is_correct"
            ),
            "pass_at_k": pass_summary,
            "maj_at_k": maj_summary,
            "weighted_maj_at_k": weighted_maj_summary,
        },
        "diversity": {
            "mean_distinct_answer_count": question_mean("distinct_answer_count"),
            "mean_distinct_answer_ratio": question_mean("distinct_answer_ratio"),
            "mean_answer_entropy": question_mean("answer_entropy"),
            "mean_normalized_answer_entropy": question_mean(
                "normalized_answer_entropy"
            ),
            "mean_majority_ratio": question_mean("majority_ratio"),
            "mean_majority_margin": question_mean("majority_margin"),
            "all_same_group_rate": question_mean("all_same"),
            "valid_gradient_group_rate": 1.0 - question_mean("all_same"),
            "invalid_answer_ratio": (
                sum(metric["invalid_answer_count"] for metric in question_metrics)
                / max(len(rollout_rows), 1)
            ),
        },
        "length": {
            "mean": mean_or_none(rollout_values("generation_length")),
            "std": std_or_none(rollout_values("generation_length")),
            "p50": quantile_or_none(rollout_values("generation_length"), 0.50),
            "p90": quantile_or_none(rollout_values("generation_length"), 0.90),
            "eos_rate": mean_or_none(rollout_values("eos_found")),
        },
        "confidence": {
            "mean_full_length": mean_or_none(
                rollout_values("mean_commit_confidence")
            ),
            "mean_pre_eos": mean_or_none(
                rollout_values("pre_eos_mean_commit_confidence")
            ),
            "pre_eos_p10": quantile_or_none(
                rollout_values("pre_eos_mean_commit_confidence"),
                0.10,
            ),
            "pre_eos_p50": quantile_or_none(
                rollout_values("pre_eos_mean_commit_confidence"),
                0.50,
            ),
            "pre_eos_p90": quantile_or_none(
                rollout_values("pre_eos_mean_commit_confidence"),
                0.90,
            ),
            "mean_pre_eos_entropy": mean_or_none(
                rollout_values("pre_eos_mean_commit_entropy")
            ),
            "mean_pre_eos_top1_top2_margin": mean_or_none(
                rollout_values("pre_eos_mean_top1_top2_margin")
            ),
            "mean_pre_eos_confidence_correct_rollouts": mean_or_none(
                correct_confidences
            ),
            "mean_pre_eos_confidence_incorrect_rollouts": mean_or_none(
                incorrect_confidences
            ),
        },
        "counts": {
            "questions": len(records),
            "rollouts": len(rollout_rows),
            "correct_rollouts": int(sum(row["is_correct"] for row in rollout_rows)),
        },
    }


def make_run_config(args, dataset_indices, world_size, eos_token_id):
    return {
        "schema_version": SCHEMA_VERSION,
        "model_path": str(Path(args.model_path).resolve()),
        "tokenizer_path": str(args.tokenizer_path or args.model_path),
        "dataset_path": args.dataset_path,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_indices": dataset_indices,
        "num_questions": args.num_questions,
        "num_rollouts": args.num_rollouts,
        "rollout_batch_size": args.rollout_batch_size,
        "k_values": list(args.k_values),
        "majority_trials": args.majority_trials,
        "temperature": args.temperature,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_size": args.block_size,
        "mask_token_id": args.mask_token_id,
        "eos_token_id": eos_token_id,
        "seed": args.seed,
        "probe_seed": args.probe_seed,
        "metric_seed": args.metric_seed,
        "world_size": world_size,
        "save_token_traces": args.save_token_traces,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on a fixed GSM8K probe set."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help=(
            "Tokenizer/base-model path. Training checkpoints usually do not "
            "contain tokenizer files, so pass the original LLaDA model path."
        ),
    )
    parser.add_argument("--output_dir", required=True)
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
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument("--majority_trials", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--mask_token_id", type=int, default=126336)
    parser.add_argument("--eos_token_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--probe_seed", type=int, default=2026)
    parser.add_argument("--metric_seed", type=int, default=314159)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--save_token_traces",
        action="store_true",
        help="Save 256-position confidence/entropy/margin arrays for every rollout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive.")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive.")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout_batch_size must be positive.")
    if args.majority_trials <= 0:
        raise ValueError("--majority_trials must be positive.")
    requested_k_values = tuple(sorted(set(args.k_values)))
    if not requested_k_values or requested_k_values[0] < 1:
        raise ValueError("--k_values must contain positive integers.")
    args.k_values = tuple(
        k for k in requested_k_values if k <= args.num_rollouts
    )
    if not args.k_values:
        raise ValueError("No --k_values remain after applying --num_rollouts.")
    removed_k_values = [
        k for k in requested_k_values if k > args.num_rollouts
    ]
    if removed_k_values:
        print(
            f"Ignoring k values larger than --num_rollouts={args.num_rollouts}: "
            f"{removed_k_values}",
            flush=True,
        )

    import torch
    import torch.distributed as torch_dist
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    from data.math import extract_answer_gsm8k
    from utils.grader import math_equal
    from utils.parser import extract_answer

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch_dist.init_process_group(backend="nccl")
        rank = torch_dist.get_rank()
        world_size = torch_dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if not torch.cuda.is_available():
        raise RuntimeError("This LLaDA probe evaluator requires CUDA.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    def barrier():
        if distributed:
            torch_dist.barrier()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "probe_summary.json"
    details_path = output_dir / "probe_details.jsonl"
    config_path = output_dir / "probe_config.json"
    rank_path = output_dir / f"probe_details.rank{rank:03d}.jsonl"

    dataset = load_dataset(
        args.dataset_path,
        args.dataset_config,
        split=args.dataset_split,
    )
    if args.num_questions > len(dataset):
        raise ValueError(
            f"Requested {args.num_questions} questions, but dataset has {len(dataset)}."
        )
    probe_rng = np.random.default_rng(args.probe_seed)
    dataset_indices = probe_rng.permutation(len(dataset))[
        : args.num_questions
    ].tolist()

    tokenizer_source = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = args.mask_token_id
    eos_token_id = args.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = 126081
    args.eos_token_id = int(eos_token_id)

    run_config = make_run_config(
        args,
        dataset_indices,
        world_size,
        eos_token_id,
    )
    if rank == 0 and not config_path.is_file():
        atomic_write_json(config_path, run_config)
    barrier()

    with config_path.open("r", encoding="utf-8") as f:
        existing_config = json.load(f)
    if existing_config != run_config:
        raise RuntimeError(
            f"Existing probe configuration differs: {config_path}. "
            "Use a new --output_dir."
        )

    if summary_path.is_file():
        if rank == 0:
            print(f"Evaluation already complete: {summary_path}")
        barrier()
        if distributed:
            torch_dist.destroy_process_group()
        return

    assigned = [
        (probe_order, dataset_index)
        for probe_order, dataset_index in enumerate(dataset_indices)
        if probe_order % world_size == rank
    ]
    existing_rows = read_jsonl(rank_path, repair=True)
    completed_indices = {row["dataset_index"] for row in existing_rows}
    pending = [item for item in assigned if item[1] not in completed_indices]

    model = None
    if pending:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        if rank == 0:
            print(f"Loading model from {args.model_path}", flush=True)
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model.eval().requires_grad_(False).to(device)

        with rank_path.open("a", encoding="utf-8") as rank_file:
            iterator = tqdm(
                pending,
                disable=rank != 0,
                desc=f"rank {rank} probe",
            )
            for probe_order, dataset_index in iterator:
                example = dataset[int(dataset_index)]
                row = evaluate_one_question(
                    model=model,
                    tokenizer=tokenizer,
                    question=example["question"],
                    ground_truth_cot=example["answer"],
                    dataset_index=int(dataset_index),
                    probe_order=int(probe_order),
                    args=args,
                    device=device,
                    eos_token_id=eos_token_id,
                    extract_answer_fn=extract_answer,
                    extract_ground_truth_fn=extract_answer_gsm8k,
                    math_equal_fn=math_equal,
                )
                rank_file.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                rank_file.flush()

    del model
    torch.cuda.empty_cache()
    barrier()

    if rank == 0:
        merged_by_index = {}
        for process_rank in range(world_size):
            process_path = output_dir / f"probe_details.rank{process_rank:03d}.jsonl"
            for row in read_jsonl(process_path):
                merged_by_index[int(row["dataset_index"])] = row

        missing = [
            dataset_index
            for dataset_index in dataset_indices
            if dataset_index not in merged_by_index
        ]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} probe questions: {missing}")

        records = sorted(
            (merged_by_index[index] for index in dataset_indices),
            key=lambda row: row["probe_order"],
        )
        write_jsonl_atomic(details_path, records)

        summary = summarize_records(records, args)
        atomic_write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved details to {details_path}", flush=True)
        print(f"Saved summary to {summary_path}", flush=True)

    barrier()
    if distributed:
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
