import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data.planning import (
    build_countdown_prompt,
    build_sudoku_prompt,
    load_planning_records,
    score_countdown_response,
    score_sudoku_response,
)
from utils.generate import generate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    records = load_planning_records(args.task, args.data_dir)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    results = []
    total_seconds = 0.0
    total_correct_cells = 0
    total_empty_cells = 0
    total_exact = 0

    for index, record in enumerate(tqdm(records, desc=f"Evaluating {args.task}")):
        if args.task == "sudoku":
            prompt_text = build_sudoku_prompt(record["puzzle"])
        else:
            prompt_text = build_countdown_prompt(record["numbers"], record["target"])

        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = tokenizer(chat, return_tensors="pt")["input_ids"].to(device)
        start = time.time()
        output = generate(
            model,
            input_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_size,
            temperature=args.temperature,
            remasking="low_confidence",
        )
        total_seconds += time.time() - start
        generated = tokenizer.decode(
            output[0, input_ids.shape[1] :], skip_special_tokens=True
        )

        if args.task == "sudoku":
            score = score_sudoku_response(
                generated, record["puzzle"], record["solution"]
            )
            total_correct_cells += score["correct_cells"]
            total_empty_cells += score["empty_cells"]
            total_exact += int(score["exact"])
            ground_truth = record["solution"]
        else:
            score = score_countdown_response(
                generated, record["numbers"], record["target"]
            )
            total_exact += int(score["correct"])
            ground_truth = [record["numbers"], record["target"]]

        results.append(
            {
                "index": index,
                "prompt": prompt_text,
                "generation": generated,
                "ground_truth": ground_truth,
                "score": score,
            }
        )

    count = len(records)
    if args.task == "sudoku":
        summary = {
            # Primary d1 metric: accuracy over originally empty cells.
            "accuracy": 100.0 * total_correct_cells / max(total_empty_cells, 1),
            "exact_accuracy": 100.0 * total_exact / max(count, 1),
            "correct_cells": total_correct_cells,
            "empty_cells": total_empty_cells,
        }
    else:
        summary = {
            "accuracy": 100.0 * total_exact / max(count, 1),
            "correct": total_exact,
        }
    summary.update(
        {
            "task": args.task,
            "num_samples": count,
            "gen_length": args.gen_length,
            "steps": args.steps,
            "block_size": args.block_size,
            "temperature": args.temperature,
            "total_seconds": total_seconds,
            "seconds_per_sample": total_seconds / max(count, 1),
        }
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dTTRL on Sudoku/Countdown")
    parser.add_argument("--task", choices=["sudoku", "countdown"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = args.gen_length
    if args.gen_length % args.block_size != 0:
        parser.error("--gen_length must be divisible by --block_size")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
