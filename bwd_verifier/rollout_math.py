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

from bwd_generate import generate
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

def process_math_dataset(model, tokenizer, device, args):
    """
    Process the MATH-500 dataset and generate rollouts for each problem
    """
    model.tokenizer = tokenizer  # Pass tokenizer to model for use in generate function
    print("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Optionally limit the number of problems for testing
    if args.max_problems > 0:
        dataset = dataset.select(range(min(args.max_problems, len(dataset))))
        print(f"Limited to {len(dataset)} problems")
    
    results = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing problems")):
        problem = example['problem']
        solution = example['solution']
        answer = example['answer']
        
        # Extract boxed answer from solution
        extracted_answer = parse_ground_truth(solution)[1]
        if not extracted_answer:
            extracted_answer = answer
        
        # Create prompt with chat template
        if args.add_solve_instruction:
            # postfix = 'Solve it step by step. Wrap the final answer in a \\boxed{}.'
            # 改成和justgrpo相同
            postfix = r'(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)'
            user_content = problem + postfix
        else:
            user_content = problem
            
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        print(f"\nProcessing problem {idx + 1}/{len(dataset)}:")
        print(f"Question: {problem}")
        print(f"prompt: {prompt}")

        # Tokenize
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
        rollouts_confidence = []
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
            
            print(f"Generated rollout {rollout_idx + 1}:\n{generated_text}\n")
            rollouts.append(generated_text)
            # rollouts_records.append(records)
            filtered_records = []
            for record in sorted(records, key=lambda x: x.get('position', 0)):
                if record.get('token_id') == 126081:
                    break  # 遇到token_id=126081时停止，不包括这条记录
                filtered_records.append(record)
            print(f"Filtered records for rollout {rollout_idx + 1} (up to token_id=126081): {len(filtered_records)} tokens")    
            # 计算置信度
            token_confidences = [record.get('confidence', 0.0) for record in filtered_records]
            if token_confidences:
                avg_confidence = sum(token_confidences) / len(token_confidences)
            else:
                avg_confidence = 0.0
            print(f"Average confidence for rollout {rollout_idx + 1}: {avg_confidence:.4f}")
            rollouts_confidence.append(avg_confidence)
        
        # Store result
        result = {
            "question": problem,
            "prompt": prompt,
            "rollouts": rollouts,
            # "rollouts_records": rollouts_records,
            "rollouts_confidence": rollouts_confidence,
            "solution": solution,
            "answer": extracted_answer
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
    parser = argparse.ArgumentParser(description='Generate solutions for MATH-500 dataset using LLaDA')
    
    # Generation parameters
    parser.add_argument('--steps', type=int, default=256,
                        help='Number of denoising steps (default: 256)')
    parser.add_argument('--gen_length', type=int, default=256,
                        help='Generated answer length in tokens (default: 256)')
    parser.add_argument('--block_length', type=int, default=32,
                        help='Block length for semi-autoregressive generation (default: 32)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.6)')
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
    parser.add_argument('--add_solve_instruction', action='store_true', default=True,
                        help='Add "Solve it step by step..." instruction to prompt (default: True)')
    parser.add_argument('--no_solve_instruction', dest='add_solve_instruction', action='store_false',
                        help='Do not add solve instruction to prompt')
    
    # Output and logging
    parser.add_argument('--output_file', type=str, default='math500_results.json',
                        help='Output JSON file name (default: math500_results.json)')
    parser.add_argument('--save_intermediate', action='store_true', default=False,
                        help='Save intermediate results (default: True)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save intermediate results every N problems (default: 10)')
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
    results = process_math_dataset(
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
    
    # Print example output
    if args.verbose and len(results) > 0:
        print("\n" + "="*50)
        print("Example output (first problem, first rollout):")
        print("="*50)
        print(f"Question: {results[0]['question'][:200]}...")
        print(f"\nRollout 1: {results[0]['rollouts'][0][:500]}...")
        print(f"\nExpected answer: {results[0]['answer']}")

if __name__ == '__main__':
    main()