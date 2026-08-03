"""Sudoku and Countdown data/reward utilities for dTTRL.

The prompts and evaluation data format follow dllm-reasoning/d1. Hidden Sudoku
solutions are removed before the training Dataset is constructed. Countdown's
target is visible in the question, but the pseudo-reward does not consult it.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from data.sampler import InfiniteSampler
from utils.distributed import get_rank, get_world_size


SUDOKU_SYSTEM_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>"""


COUNTDOWN_SYSTEM_PROMPT = """Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and /, but each number must be used exactly once.

Respond in this exact format:
<reasoning>
Your step-by-step reasoning
</reasoning>
<answer>
[final expression only, without an equals sign or the target]
</answer>"""


def build_sudoku_prompt(puzzle: str) -> str:
    return f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {puzzle}\n"


def build_countdown_prompt(numbers: Sequence[int], target: int) -> str:
    return (
        f"{COUNTDOWN_SYSTEM_PROMPT}\n\n"
        f"Numbers: {[int(n) for n in numbers]}\nTarget: {int(target)}\n"
    )


def load_planning_records(task: str, data_dir: str | Path = "dataset") -> list[dict]:
    """Load the d1 Sudoku/Countdown evaluation records from local files."""
    data_dir = Path(data_dir)
    if task == "sudoku":
        path = data_dir / "4x4_test_sudoku.csv"
        _require_file(path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        records = []
        for row in rows:
            puzzle = str(row["Puzzle"]).zfill(16)
            solution = str(row["Solution"]).zfill(16)
            if len(puzzle) != 16 or len(solution) != 16:
                raise ValueError(f"Invalid Sudoku row in {path}: {row}")
            records.append({"puzzle": puzzle, "solution": solution})
        return records

    if task == "countdown":
        path = data_dir / "countdown_cd3_test.jsonl"
        _require_file(path)
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_numbers = row["input"]
                if isinstance(raw_numbers, str):
                    numbers = [int(x.strip()) for x in raw_numbers.split(",")]
                else:
                    numbers = [int(x) for x in raw_numbers]
                records.append({"numbers": numbers, "target": int(row["output"])})
        return records

    raise ValueError(f"Unsupported task: {task!r}; expected 'sudoku' or 'countdown'.")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: bash download_d1_planning_data.sh {path.parent}"
        )


def _collate_sudoku(batch: list[dict]) -> dict:
    puzzles = [str(item["puzzle"]) for item in batch]
    return {
        "problems": [build_sudoku_prompt(puzzle) for puzzle in puzzles],
        "puzzles": puzzles,
    }


def _collate_countdown(batch: list[dict]) -> dict:
    numbers = [[int(x) for x in item["numbers"]] for item in batch]
    targets = [int(item["target"]) for item in batch]
    return {
        "problems": [
            build_countdown_prompt(item_numbers, target)
            for item_numbers, target in zip(numbers, targets)
        ],
        "numbers": numbers,
    }


def load_planning_dataset_and_reward(
    task: str,
    data_dir: str | Path = "dataset",
    batch_size: int = 1,
    num_workers: int = 4,
    seed: int = 112,
) -> tuple[DataLoader, Callable]:
    """Return an infinite distributed dataloader and dTTRL pseudo-reward."""
    records = load_planning_records(task, data_dir)
    if task == "sudoku":
        # Do not place hidden solutions in the training Dataset or batch.
        training_records = [{"puzzle": item["puzzle"]} for item in records]
    else:
        # Countdown's target is part of the user-visible problem statement.
        training_records = records
    dataset = Dataset.from_list(training_records).shuffle(seed=seed)
    sampler = InfiniteSampler(
        dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
    )
    collate_fn = _collate_sudoku if task == "sudoku" else _collate_countdown
    reward_fn = reward_ttrl_sudoku if task == "sudoku" else reward_ttrl_countdown
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    return dataloader, reward_fn


def _extract_last_answer_tag(text: str) -> Optional[str]:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def _extract_last_boxed(text: str) -> Optional[str]:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    pos = start + len(r"\boxed{")
    depth = 1
    for index in range(pos, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[pos:index].strip()
    return None


def extract_sudoku_digits(text: str) -> Optional[str]:
    """Extract digits from the final answer tag, with a conservative fallback."""
    answer = _extract_last_answer_tag(text)
    if answer is not None:
        digits = "".join(re.findall(r"\d", answer))
        if digits:
            return digits
    matches = re.findall(r"(?<!\d)([1-4](?:\s*[1-4]){15})(?!\d)", text)
    if matches:
        return re.sub(r"\s", "", matches[-1])
    return None


def canonicalize_sudoku_response(text: str, puzzle: str) -> Optional[str]:
    """Build a vote key without using the ground-truth Sudoku solution."""
    digits = extract_sudoku_digits(text)
    if digits is None or len(digits) != 16 or any(ch not in "1234" for ch in digits):
        return None
    if len(puzzle) != 16:
        return None
    # Reject outputs that overwrite a given clue. Sudoku constraints themselves are
    # not used here, so the pseudo-label is still selected by rollout voting.
    if any(clue != "0" and digits[i] != clue for i, clue in enumerate(puzzle)):
        return None
    return digits


def _normalize_countdown_text(expression: str) -> str:
    expression = expression.strip()
    boxed = _extract_last_boxed(expression)
    if boxed is not None:
        expression = boxed
    expression = (
        expression.replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace(r"\div", "/")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
    # Handle simple LaTeX fractions, which are common in model outputs.
    fraction_pattern = re.compile(r"\\(?:d?frac|tfrac)\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    while previous != expression:
        previous = expression
        expression = fraction_pattern.sub(r"(\1)/(\2)", expression)
    if "=" in expression:
        expression = expression.split("=", 1)[0]
    return expression.strip().strip("$` ")


def extract_countdown_expression(text: str) -> Optional[str]:
    answer = _extract_last_answer_tag(text)
    if answer is None:
        answer = _extract_last_boxed(text)
    if answer is None:
        return None
    expression = _normalize_countdown_text(answer)
    return expression or None


def _eval_arithmetic(node: ast.AST, used_numbers: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic(node.body, used_numbers)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        if node.value < 0:
            raise ValueError("Negative literals must be formed with unary minus")
        used_numbers.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic(node.operand, used_numbers)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_arithmetic(node.left, used_numbers)
        right = _eval_arithmetic(node.right, used_numbers)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def evaluate_countdown_expression(
    expression: str, available_numbers: Sequence[int]
) -> Optional[Fraction]:
    """Safely evaluate an expression and enforce exact number usage."""
    try:
        tree = ast.parse(expression, mode="eval")
        used_numbers: list[int] = []
        value = _eval_arithmetic(tree, used_numbers)
        if sorted(used_numbers) != sorted(int(x) for x in available_numbers):
            return None
        return value
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def canonicalize_countdown_response(
    text: str, available_numbers: Sequence[int]
) -> Optional[str]:
    """Group different valid expressions by their exact numerical result."""
    expression = extract_countdown_expression(text)
    if expression is None:
        return None
    value = evaluate_countdown_expression(expression, available_numbers)
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _confidence_weighted_rewards(
    keys: Sequence[Optional[str]],
    confidences: Optional[Sequence[float]],
    num_generations: int,
    device: torch.device,
    task: str,
) -> torch.Tensor:
    if len(keys) % num_generations != 0:
        raise ValueError(
            f"Received {len(keys)} responses, which is not divisible by group size {num_generations}."
        )
    if confidences is not None and len(confidences) != len(keys):
        raise ValueError("confidences and responses must have the same length")

    rewards = torch.zeros(len(keys), dtype=torch.float32, device=device)
    num_problems = len(keys) // num_generations
    for problem_idx in range(num_problems):
        start = problem_idx * num_generations
        end = start + num_generations
        group_keys = list(keys[start:end])
        if confidences is None:
            group_weights = [1.0] * num_generations
        else:
            group_weights = []
            for confidence in confidences[start:end]:
                weight = float(confidence)
                group_weights.append(weight if math.isfinite(weight) and weight > 0 else 0.0)

        weighted_scores: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        first_seen: dict[str, int] = {}
        for local_idx, (key, weight) in enumerate(zip(group_keys, group_weights)):
            if key is None:
                continue
            weighted_scores[key] += weight
            counts[key] += 1
            first_seen.setdefault(key, local_idx)
        if not weighted_scores:
            print(f"[dTTRL:{task}] no valid answers in rollout group", flush=True)
            continue
        if sum(weighted_scores.values()) == 0:
            weighted_scores = {key: float(count) for key, count in counts.items()}

        winner = max(
            weighted_scores,
            key=lambda key: (weighted_scores[key], counts[key], -first_seen[key]),
        )
        for local_idx, key in enumerate(group_keys):
            if key == winner:
                rewards[start + local_idx] = 1.0
        print(
            f"[dTTRL:{task}] valid={sum(key is not None for key in group_keys)}/{num_generations} "
            f"distinct={len(weighted_scores)} winner={winner!r} "
            f"winner_count={counts[winner]}",
            flush=True,
        )
    return rewards


def reward_ttrl_sudoku(
    batch: dict,
    responses: Sequence[str],
    num_generations: int,
    device: torch.device,
    confidences: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    puzzles = list(batch["puzzles"])
    keys: list[Optional[str]] = []
    for problem_idx, puzzle in enumerate(puzzles):
        start = problem_idx * num_generations
        end = start + num_generations
        keys.extend(canonicalize_sudoku_response(text, puzzle) for text in responses[start:end])
    return _confidence_weighted_rewards(keys, confidences, num_generations, device, "sudoku")


def reward_ttrl_countdown(
    batch: dict,
    responses: Sequence[str],
    num_generations: int,
    device: torch.device,
    confidences: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    number_groups = list(batch["numbers"])
    keys: list[Optional[str]] = []
    for problem_idx, numbers in enumerate(number_groups):
        start = problem_idx * num_generations
        end = start + num_generations
        keys.extend(canonicalize_countdown_response(text, numbers) for text in responses[start:end])
    return _confidence_weighted_rewards(keys, confidences, num_generations, device, "countdown")


def score_sudoku_response(text: str, puzzle: str, solution: str) -> dict:
    """Match d1's empty-cell accuracy and additionally report exact accuracy."""
    predicted = extract_sudoku_digits(text) or ""
    predicted = (predicted + "0" * 16)[:16]
    empty_indices = [i for i, clue in enumerate(puzzle) if clue == "0"]
    correct_cells = sum(predicted[i] == solution[i] for i in empty_indices)
    cell_accuracy = correct_cells / len(empty_indices) if empty_indices else 0.0
    return {
        "prediction": predicted,
        "correct_cells": correct_cells,
        "empty_cells": len(empty_indices),
        "cell_accuracy": cell_accuracy,
        "exact": predicted == solution,
    }


def score_countdown_response(text: str, numbers: Sequence[int], target: int) -> dict:
    expression = extract_countdown_expression(text)
    value = (
        evaluate_countdown_expression(expression, numbers)
        if expression is not None
        else None
    )
    return {
        "expression": expression,
        "value": str(value) if value is not None else None,
        "correct": value == Fraction(int(target)),
    }
