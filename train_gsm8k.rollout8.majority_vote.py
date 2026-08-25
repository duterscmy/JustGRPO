"""GSM8K dTTRL with uniform majority voting and confidence-only logging.

This is a non-invasive entry point built on top of
``train_gsm8k.rollout8.debug_collapse.py``.  It intentionally keeps rollout
confidence collection and the existing analysis log format, but confidence is
never used to select the majority answer or construct rewards.

The base training file is loaded at runtime and is not modified.
"""

from __future__ import annotations

import runpy
import sys
from collections import defaultdict
from pathlib import Path

import torch

import grpo
from data.math import (
    extract_answer,
    extract_answer_gsm8k,
    math_equal,
    parse_ground_truth,
)
from utils.generate import generate_with_confidence


BASE_TRAIN_SCRIPT = Path(__file__).with_name(
    "train_gsm8k.rollout8.debug_collapse.py"
)


def reward_ttrl_uniform_with_confidence_log(
    batch,
    responses,
    num_generations,
    device,
    confidences,
):
    """Use uniform majority voting while preserving the old confidence logs.

    ``confidences`` is printed for diagnostics only.  Every normalized answer
    receives one vote, irrespective of its rollout confidence.
    """
    # Keep this exact prefix for compatibility with the existing plot parser.
    print("confidences {}".format(confidences))

    ground_truth_cot = list(batch["answers"])[0]
    if "####" in ground_truth_cot:
        answer = extract_answer_gsm8k(ground_truth_cot)
    else:
        answer = parse_ground_truth(ground_truth_cot)[1]

    print("======correct answer: {}======".format(answer))

    num_problems = len(responses) // num_generations
    rewards = torch.zeros(len(responses), device=device)

    for problem_idx in range(num_problems):
        start_idx = problem_idx * num_generations
        end_idx = start_idx + num_generations
        problem_responses = responses[start_idx:end_idx]

        print("============ROLLOUT==========")
        extracted_answers_raw = []
        for response in problem_responses:
            extracted = extract_answer(response)
            extracted_answers_raw.append(extracted)
            print(response)
            print(extracted)
            print("==================")

        # Preserve the same math-equivalence normalization used by reward_ttrl.
        unique_answers = []
        for extracted in extracted_answers_raw:
            if extracted not in unique_answers:
                unique_answers.append(extracted)

        canonical_representatives = []
        answer_to_canonical = {}
        for extracted in unique_answers:
            for representative in canonical_representatives:
                if math_equal(extracted, representative):
                    answer_to_canonical[extracted] = representative
                    break
            else:
                canonical_representatives.append(extracted)
                answer_to_canonical[extracted] = extracted

        normalized_answers = [
            answer_to_canonical[extracted]
            for extracted in extracted_answers_raw
        ]

        normalized_ground_truth = answer
        for representative in canonical_representatives:
            if math_equal(answer, representative):
                normalized_ground_truth = representative
                break

        if not normalized_answers:
            continue

        # Uniform majority: confidence does not enter this counter.
        vote_counter = defaultdict(int)
        for normalized_answer in normalized_answers:
            vote_counter[normalized_answer] += 1

        majority_answer = max(vote_counter.items(), key=lambda item: item[1])[0]

        all_answer_num = len(extracted_answers_raw)
        distinct_answer_num = len(vote_counter)
        distinct_answer_ratio = distinct_answer_num / all_answer_num
        best_answer_ratio = vote_counter[majority_answer] / all_answer_num
        correct_answer_number = sum(
            normalized_answer == normalized_ground_truth
            for normalized_answer in normalized_answers
        )
        best_is_correct = int(majority_answer == normalized_ground_truth)

        print(
            f"==========MAJORITY: {majority_answer} (uniform)==========="
        )
        print(
            f"diversity| distinct_answer_num: {distinct_answer_num} | "
            f"all_answer_num: {all_answer_num} | "
            f"distinct_answer_ratio: {distinct_answer_ratio:.2f} | "
            f"best_answer_ratio: {best_answer_ratio:.2f} | "
            f"correct_answer_number: {correct_answer_number} | "
            f"best_is_correct: {best_is_correct} | "
            f"extracted_answers: {extracted_answers_raw} | "
            f"normalized_answers: {normalized_answers} | "
            f"majority_answer: {majority_answer} | "
            f"ground_truth_answer: {answer}",
            flush=True,
        )

        for index, normalized_answer in enumerate(normalized_answers):
            if normalized_answer == majority_answer:
                rewards[start_idx + index] = 1.0

    return rewards


@torch.no_grad()
def sample_with_logged_confidence_uniform_vote(
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
    """Generate with confidence diagnostics, then apply uniform voting."""
    if apply_chat_template:
        prompts = tokenizer.apply_chat_template(
            [[{"role": "user", "content": problem}] for problem in batch["problems"]],
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
    print("=======block size:{}======".format(block_size))

    for _ in range(repeat_time):
        repeated_prompt_ids = prompt_ids.repeat(num_generations, 1)
        generated_ids, confidence_values = generate_with_confidence(
            model=model,
            prompt=repeated_prompt_ids,
            steps=steps,
            gen_length=gen_length,
            temperature=temperature,
            block_length=block_size,
        )

        completion_lengths = grpo.get_completion_lengths(
            generated_ids,
            prompt_len=prompt_ids.shape[1],
        )
        print(f"avg_gen_length: {completion_lengths.float().mean():.1f}")

        for generated_id, confidence in zip(generated_ids, confidence_values):
            generated_with_confidence.append((generated_id, confidence))

    # Preserve the existing auxiliary line as well as the exact
    # ``confidences [...]`` line emitted by the reward function below.
    print(
        "confidence list:",
        [confidence for _, confidence in generated_with_confidence],
    )

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
    all_confidences = [
        confidence for _, confidence in generated_with_confidence
    ]
    print(all_generated_ids.size())

    responses = tokenizer.batch_decode(
        all_generated_ids,
        skip_special_tokens=True,
    )
    rewards = reward_ttrl_uniform_with_confidence_log(
        batch=batch,
        responses=responses,
        num_generations=num_generations * repeat_time,
        device=device,
        confidences=all_confidences,
    ).float()

    return {
        "generated_ids": all_generated_ids,
        "prompt_len": prompt_ids.shape[1],
        "rewards": rewards,
    }


def main():
    if not BASE_TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Required base training file was not found: {BASE_TRAIN_SCRIPT}"
        )

    if "--dynamic_sampling" in sys.argv[1:]:
        raise SystemExit(
            "This entry point is intentionally for no-Dynamic-Sampling "
            "experiments; remove --dynamic_sampling."
        )

    # The base script imports this symbol from grpo.  Rebinding it only in the
    # current process keeps every existing source file unchanged while reusing
    # the established optimizer, diagnostics, checkpoint and logging logic.
    grpo.sample_with_weighted_confidence = (
        sample_with_logged_confidence_uniform_vote
    )

    print("Voting mode: uniform majority (confidence is logging-only)")
    print("Dynamic sampling: forced off by this entry point")
    runpy.run_path(str(BASE_TRAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
