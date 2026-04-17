import torch
import json
import re
import random
import argparse
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
import os

from generate import generate
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parser import extract_answer, parse_ground_truth

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_arc_answer(pred_str):
    """
    从模型的输出中提取 ARC 选择题答案 (A/B/C/D)
    
    Args:
        pred_str: 模型的输出字符串
    
    Returns:
        提取的答案字母 (A/B/C/D)，如果未找到返回 None
    """
    if not pred_str or not isinstance(pred_str, str):
        return None
    
    # 匹配 "the answer is (X)" 格式，找最后一个
    pattern = r'[Tt]he\s+answer\s+is\s+\(([A-D])\)'
    matches = re.findall(pattern, pred_str)
    
    if matches:
        return matches[-1]
    
    return None

def process_arc_dataset(model, tokenizer, device, args):
    """
    Process the ARC-Challenge dataset and generate rollouts for each problem
    """
    print("Loading ARC-Challenge dataset...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=args.split)
    
    # Optionally limit the number of problems for testing
    if args.max_problems > 0:
        dataset = dataset.select(range(min(args.max_problems, len(dataset))))
        print(f"Limited to {len(dataset)} problems")
    
    results = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing problems")):
        question = example['question'].strip()
        choices = example['choices']
        choice_texts = choices['text']
        answer_key = example['answerKey']
        
        # 构建 prompt 模板（不使用 chat template）
        prompt = f"Q: {question}\n(A) {choice_texts[0]} (B) {choice_texts[1]} (C) {choice_texts[2]}"
        
        # 如果有第4个选项（D）
        if len(choice_texts) > 3:
            prompt += f" (D) {choice_texts[3]}"
        
        prompt += "\nA: Let's think step by step."
        
        print(f"\nProcessing problem {idx + 1}/{len(dataset)}:")
        print(f"Question: {question}")
        print(f"Prompt: {prompt}")
        print(f"Answer Key: {answer_key}")

        # Tokenize (no chat template, just direct encoding)
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Generate multiple rollouts
        rollouts = []
        rollouts_records = []
        extracted_answers = []
        
        for rollout_idx in range(args.num_rollouts):
            if args.verbose:
                print(f"\nGenerating rollout {rollout_idx + 1}/{args.num_rollouts} for problem {idx + 1}")
            
            with torch.no_grad():
                out, records = generate(
                    model, 
                    input_ids, 
                    attention_mask, 
                    steps=args.steps, 
                    gen_length=args.gen_length, 
                    block_length=args.block_length, 
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale, 
                    remasking=args.remasking
                )
            
            generated_text = tokenizer.batch_decode(
                out[:, input_ids.shape[1]:], 
                skip_special_tokens=True
            )[0]
            
            # 提取答案
            extracted_answer = extract_arc_answer(generated_text)
            
            print(f"Generated rollout {rollout_idx + 1}:\n{generated_text}\n")
            print(f"Extracted answer: {extracted_answer}")
            
            rollouts.append(generated_text)
            rollouts_records.append(records)
            extracted_answers.append(extracted_answer)
        
        # 计算准确率
        correct_count = sum(1 for ans in extracted_answers if ans == answer_key)
        accuracy = correct_count / len(extracted_answers) if extracted_answers else 0
        
        # 多样性统计
        unique_answers = set([a for a in extracted_answers if a is not None])
        distinct_answer_num = len(unique_answers)
        all_answer_num = len(extracted_answers)
        
        # 多数投票答案
        from collections import Counter
        if extracted_answers:
            counter = Counter([a for a in extracted_answers if a is not None])
            if counter:
                majority_answer = counter.most_common(1)[0][0]
                best_answer_ratio = counter[majority_answer] / all_answer_num
            else:
                majority_answer = None
                best_answer_ratio = 0
        else:
            majority_answer = None
            best_answer_ratio = 0
        
        best_is_correct = 1 if majority_answer == answer_key else 0
        
        # 打印统计信息
        print(f"\n=== Problem {idx + 1} Summary ===")
        print(f"Extracted answers: {extracted_answers}")
        print(f"Ground truth: {answer_key}")
        print(f"Majority answer: {majority_answer}")
        print(f"Best is correct: {best_is_correct}")
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(extracted_answers)})")
        print(f"Distinct answers: {distinct_answer_num}")
        print(f"Best answer ratio: {best_answer_ratio:.2f}")
        
        # 输出 diversity 格式（与之前代码兼容）
        print(f"diversity| distinct_answer_num: {distinct_answer_num} | all_answer_num: {all_answer_num} | best_answer_ratio: {best_answer_ratio:.2f} | correct_answer_number: {correct_count} | best_is_correct: {best_is_correct} | extracted_answers: {extracted_answers} | majority_answer: {majority_answer} | ground_truth_answer: {answer_key}")
        
        # Store result
        result = {
            "question": question,
            "choices": choice_texts,
            "prompt": prompt,
            "rollouts": rollouts,
            "rollouts_records": rollouts_records,
            "extracted_answers": extracted_answers,
            "answer_key": answer_key,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "distinct_answer_num": distinct_answer_num,
            "majority_answer": majority_answer,
            "best_is_correct": best_is_correct
        }
        results.append(result)
        
        # Save intermediate results
        if args.save_intermediate and (idx + 1) % args.save_every == 0:
            intermediate_file = args.output_file.replace('.json', f'_intermediate_{idx+1}.json')
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if args.verbose:
                print(f"\nSaved intermediate results to {intermediate_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate solutions for ARC-Challenge dataset using LLaDA')
    
    # Generation parameters
    parser.add_argument('--steps', type=int, default=64,
                        help='Number of denoising steps (default: 64)')
    parser.add_argument('--gen_length', type=int, default=64,
                        help='Generated answer length in tokens (default: 64)')
    parser.add_argument('--block_length', type=int, default=8,
                        help='Block length for semi-autoregressive generation (default: 8)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--cfg_scale', type=float, default=0.0,
                        help='Classifier-free guidance scale (default: 0.0)')
    parser.add_argument('--remasking', type=str, default='low_confidence',
                        choices=['low_confidence', 'random'],
                        help='Remasking strategy (default: low_confidence)')
    
    # Dataset and rollout parameters
    parser.add_argument('--num_rollouts', type=int, default=8,
                        help='Number of rollouts to generate per problem (default: 8)')
    parser.add_argument('--max_problems', type=int, default=-1,
                        help='Maximum number of problems to process (-1 for all, default: -1)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to use (default: test)')
    
    # Output and logging
    parser.add_argument('--output_file', type=str, default='arc_results.json',
                        help='Output JSON file name (default: arc_results.json)')
    parser.add_argument('--save_intermediate', action='store_true', default=False,
                        help='Save intermediate results (default: False)')
    parser.add_argument('--save_every', type=int, default=200,
                        help='Save intermediate results every N problems (default: 200)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print detailed progress information (default: False)')
    
    # Model and device
    parser.add_argument('--model_path', type=str, 
                        default='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',
                        help='Path to LLaDA model (default: specified path)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")
    
    # Adjust device if CUDA not available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = args.device
    print(f"Using device: {device}")
    print(f"Generation parameters: steps={args.steps}, gen_length={args.gen_length}, "
          f"block_length={args.block_length}, temperature={args.temperature}, "
          f"cfg_scale={args.cfg_scale}, remasking={args.remasking}")
    print(f"Rollouts per problem: {args.num_rollouts}")
    
    # Load model and tokenizer
    print(f"Loading LLaDA model from {args.model_path}...")
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    
    # Set padding side to left (required for generation)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
        print("Set padding_side to 'left'")
    
    # Verify padding token is not mask token
    assert tokenizer.pad_token_id != 126336, "Padding token ID conflicts with mask token ID"
    
    # Process dataset
    results = process_arc_dataset(
        model=model,
        tokenizer=tokenizer,
        device=device,
        args=args
    )
    
    # Save final results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {args.output_file}")
    print(f"Processed {len(results)} problems")
    print(f"Total rollouts generated: {len(results) * args.num_rollouts}")
    
    # Print overall statistics
    if len(results) > 0:
        total_accuracy = sum(r['accuracy'] for r in results) / len(results)
        total_best_is_correct = sum(r['best_is_correct'] for r in results) / len(results)
        avg_distinct = sum(r['distinct_answer_num'] for r in results) / len(results)
        
        print("\n" + "="*50)
        print("Overall Statistics:")
        print("="*50)
        print(f"Average rollout accuracy: {total_accuracy:.2%}")
        print(f"Voting accuracy (best_is_correct): {total_best_is_correct:.2%}")
        print(f"Average distinct answers: {avg_distinct:.2f}")
        
        # Print example output
        if args.verbose:
            print("\n" + "="*50)
            print("Example output (first problem, first rollout):")
            print("="*50)
            print(f"Question: {results[0]['question'][:200]}...")
            print(f"\nRollout 1: {results[0]['rollouts'][0][:500]}...")
            print(f"\nExpected answer: {results[0]['answer_key']}")

if __name__ == '__main__':
    main()