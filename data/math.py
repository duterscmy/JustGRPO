"""
GSM8K Dataset Loader for LLaDOU Training.

Provides dataloader and reward function for GSM8K math problems.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.distributed import get_rank, get_world_size
from data.sampler import InfiniteSampler
from utils.grader import math_equal
from utils.parser import extract_answer, parse_ground_truth, extract_arc_answer


def collate_fn_gsm8k(batch):
    """Collate function for GSM8K dataset."""
    problems = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    return {"problems": problems, "answers": answers}

def collate_fn_math(batch):
    """Collate function for MATH dataset."""
    problems = []
    answers = []
    instruct = r"(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)"
    for item in batch:
        problems.append(item['problem'] + instruct)
        answers.append(item['solution'])
    return {"problems": problems, "answers": answers}

def collate_fn_aime2024(batch):
    """Collate function for GSM8K dataset."""
    problems = [item['Problem'] for item in batch]
    answers = [item['Solution'] for item in batch]
    return {"problems": problems, "answers": answers}

def collate_fn_arc(batch):
    """Collate function for ARC dataset."""
    problems = []
    answers = []
    for item in batch:
        question = item['question'].strip()
        choices = item['choices']
        choice_texts = choices['text']
        
        # 构建 prompt 模板
        prompt = f"Q: {question}\n(A) {choice_texts[0]} (B) {choice_texts[1]} (C) {choice_texts[2]}"
        
        # 如果有第4个选项（D）
        if len(choice_texts) > 3:
            prompt += f" (D) {choice_texts[3]}"
        
        prompt += "\nA: Let's think step by step."
        problems.append(prompt)
        answers.append(item['answerKey'])
    print("answers: {}".format(answers))
    return {"problems": problems, "answers": answers}

def extract_answer_gsm8k(answer: str):
    """Extract the final answer from GSM8K format (after ####)."""
    return answer.split('####')[-1].strip()

def reward_gsm8k(batch, responses, num_generations, device):
    """
    Compute reward for GSM8K responses.
    
    Args:
        batch: Batch containing ground truth answers
        responses: Model generated responses
        num_generations: Number of generations per problem
        device: Torch device
    
    Returns:
        Tensor of rewards (+1 for correct, -1 for incorrect)
    """
    answers = batch['answers'] * num_generations
    
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]
    
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if math_equal(ans, res):
            rewards[i] = 1.0
        else:
            rewards[i] = -1.0
    
    return rewards

def reward_MATH(batch, responses, num_generations, device):
    """Compute reward for MATH responses (+1 correct, -1 incorrect)."""
    answers = batch['answers'] * num_generations
    ext_ans = [extract_answer(ans) for ans in answers]
    ext_res = [parse_ground_truth(res)[1] for res in responses]
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if math_equal(ans, res, timeout=True):
            rewards[i] = 1.0
        else:
            rewards[i] = -1.0
    return rewards

def reward_ttrl_old(batch, responses, num_generations, device):
    """
    Compute reward for GSM8K responses using TTRL's majority voting method.
    
    Args:
        batch: Batch containing problems (but NOT using ground truth answers for reward)
        responses: Model generated responses
        num_generations: Number of generations per problem
        device: Torch device
    
    Returns:
        Tensor of rewards (+1 for matching majority vote, -1 for not matching)
    """
    from collections import Counter
    
    ground_truth_cot = list(batch['answers'])[0]
    if "####" in ground_truth_cot:
        answer = extract_answer_gsm8k(ground_truth_cot)
    else:
        answer = parse_ground_truth(ground_truth_cot)[1]

    print("======correct answer: {}======".format(answer))
    num_problems = len(responses) // num_generations
    rewards = torch.zeros(len(responses), device=device)
    
    for problem_idx in range(num_problems):
        # 获取当前问题的所有生成结果
        start_idx = problem_idx * num_generations
        end_idx = start_idx + num_generations
        problem_responses = responses[start_idx:end_idx]
        
        # 提取所有答案
        print("============ROLLOUT==========")
        extracted_answers = []
        for resp in problem_responses:
            ans = extract_answer(resp)
            extracted_answers.append(ans)
            print(resp)
            print(ans)
            print("==================")
        
        # 多数投票：找出出现频率最高的答案作为伪标签
        if extracted_answers:
            counter = Counter(extracted_answers)
            majority_answer = counter.most_common(1)[0][0]
            print("==========MAJORITY: {}===========".format(majority_answer))

            # 计算多样性统计
            distinct_answer_num = len(counter)
            all_answer_num = len(extracted_answers)
            distinct_answer_ratio = distinct_answer_num / all_answer_num
            best_answer_ratio = counter[majority_answer] / all_answer_num
            
            # 计算正确答案数量
            correct_answer_number = sum(1 for ans in extracted_answers if ans == answer)
            
            # 判断最佳答案是否等于正确答案
            best_is_correct = 1 if majority_answer == answer else 0
            
            # 输出多样性统计和正确答案数量（特定格式）
            print(f"diversity| distinct_answer_num: {distinct_answer_num} | all_answer_num: {all_answer_num} | distinct_answer_ratio: {distinct_answer_ratio:.2f} | best_answer_ratio: {best_answer_ratio:.2f} | correct_answer_number: {correct_answer_number} | best_is_correct: {best_is_correct} | extracted_answers: {extracted_answers} | majority_answer: {majority_answer} | ground_truth_answer: {answer}", flush=True)

            # 根据是否匹配多数投票结果分配奖励
            for i, ans in enumerate(extracted_answers):
                if ans == majority_answer:
                    rewards[start_idx + i] = 1.0  # 匹配多数投票结果
                # else:
                #     rewards[start_idx + i] = -1.0  # 不匹配多数投票结果
    
    return rewards


def reward_ttrl(batch, responses, num_generations, device):
    """
    Compute reward for GSM8K responses using TTRL's majority voting method.
    
    Args:
        batch: Batch containing problems (but NOT using ground truth answers for reward)
        responses: Model generated responses
        num_generations: Number of generations per problem
        device: Torch device
        math_equal: Function to check if two answers are mathematically equivalent.
                    If None, falls back to exact string matching.
    
    Returns:
        Tensor of rewards (+1 for matching majority vote, -1 for not matching)
    """
    from collections import defaultdict
    
    ground_truth_cot = list(batch['answers'])[0]
    if "####" in ground_truth_cot:
        answer = extract_answer_gsm8k(ground_truth_cot)
    else:
        answer = parse_ground_truth(ground_truth_cot)[1]

    print("======correct answer: {}======".format(answer))
    num_problems = len(responses) // num_generations
    rewards = torch.zeros(len(responses), device=device)
    
    for problem_idx in range(num_problems):
        # 获取当前问题的所有生成结果
        start_idx = problem_idx * num_generations
        end_idx = start_idx + num_generations
        problem_responses = responses[start_idx:end_idx]
        
        # 提取所有答案
        print("============ROLLOUT==========")
        extracted_answers_raw = []
        for resp in problem_responses:
            ans = extract_answer(resp)
            extracted_answers_raw.append(ans)
            print(resp)
            print(ans)
            print("==================")
        
        # ========== 基于 math_equal 进行答案归一化 ==========
        # 收集所有唯一的原始答案
        unique_answers = []
        for ans in extracted_answers_raw:
            if ans not in unique_answers:
                unique_answers.append(ans)
        
        # 建立等价关系
        canonical_representatives = []  # 每个等价类的代表元
        ans_to_canonical = {}           # 原始答案 -> 规范答案
        
        for ans in unique_answers:
            found = False
            for rep in canonical_representatives:
                if math_equal(ans, rep):
                    ans_to_canonical[ans] = rep
                    found = True
                    break
            if not found:
                canonical_representatives.append(ans)
                ans_to_canonical[ans] = ans
        
        # 归一化答案列表
        normalized_answers = [ans_to_canonical[ans] for ans in extracted_answers_raw]
        
        # 同时归一化 ground truth answer
        # 找到与 answer 等价的规范答案
        normalized_answer = answer
        for rep in canonical_representatives:
            if math_equal(answer, rep):
                normalized_answer = rep
                break
        
        # 对归一化后的答案进行投票
        if normalized_answers:
            # 统计归一化后的答案
            counter = defaultdict(int)
            for norm_ans in normalized_answers:
                counter[norm_ans] += 1
            
            # 找出出现频率最高的答案
            majority_norm_answer = max(counter.items(), key=lambda x: x[1])[0]
            majority_answer = majority_norm_answer
            
            # 计算多样性统计
            distinct_answer_num_norm = len(counter)
            all_answer_num = len(extracted_answers_raw)
            distinct_answer_ratio = distinct_answer_num_norm / all_answer_num
            best_answer_ratio = counter[majority_norm_answer] / all_answer_num
            
            # 计算正确答案数量：直接比较归一化后的答案
            correct_answer_number = sum(1 for norm_ans in normalized_answers if norm_ans == normalized_answer)
            
            # 判断最佳答案是否等于正确答案
            best_is_correct = 1 if majority_norm_answer == normalized_answer else 0
            
            print(f"==========MAJORITY: {majority_answer} (norm)===========")
            
            # 输出多样性统计
            print(f"diversity| distinct_answer_num: {distinct_answer_num_norm} | all_answer_num: {all_answer_num} | distinct_answer_ratio: {distinct_answer_ratio:.2f} | best_answer_ratio: {best_answer_ratio:.2f} | correct_answer_number: {correct_answer_number} | best_is_correct: {best_is_correct} | extracted_answers: {extracted_answers_raw} | normalized_answers: {normalized_answers} | majority_answer: {majority_answer} | ground_truth_answer: {answer}", flush=True)

            # 根据是否匹配多数投票结果分配奖励
            for i, norm_ans in enumerate(normalized_answers):
                if norm_ans == majority_norm_answer:
                    rewards[start_idx + i] = 1.0
                # else:
                #     rewards[start_idx + i] = -1.0
    
    return rewards


def reward_ttrl_arc(batch, responses, num_generations, device):
    """
    Compute reward for ARC responses using TTRL's majority voting method.
    
    Args:
        batch: Batch containing problems (but NOT using ground truth answers for reward)
        responses: Model generated responses
        num_generations: Number of generations per problem
        device: Torch device
    
    Returns:
        Tensor of rewards (+1 for matching majority vote, -1 for not matching)
    """
    from collections import Counter
    
    answer = list(batch['answers'])[0]
    print("======correct answer: {}======".format(answer))

    num_problems = len(responses) // num_generations
    rewards = torch.zeros(len(responses), device=device)
    
    for problem_idx in range(num_problems):
        # 获取当前问题的所有生成结果
        start_idx = problem_idx * num_generations
        end_idx = start_idx + num_generations
        problem_responses = responses[start_idx:end_idx]
        
        # 提取所有答案
        print("============ROLLOUT==========")
        extracted_answers = []
        for resp in problem_responses:
            ans = extract_arc_answer(resp)
            extracted_answers.append(ans)
            print(resp)
            print(ans)
            print("==================")
        
        # 多数投票：找出出现频率最高的答案作为伪标签
        if extracted_answers:
            counter = Counter(extracted_answers)
            majority_answer = counter.most_common(1)[0][0]
            print("==========MAJORITY: {}===========".format(majority_answer))

            # 计算多样性统计
            distinct_answer_num = len(counter)
            all_answer_num = len(extracted_answers)
            distinct_answer_ratio = distinct_answer_num / all_answer_num
            best_answer_ratio = counter[majority_answer] / all_answer_num
            
            # 计算正确答案数量
            correct_answer_number = sum(1 for ans in extracted_answers if ans == answer)
            
            # 判断最佳答案是否等于正确答案
            best_is_correct = 1 if majority_answer == answer else 0
            
            # 输出多样性统计和正确答案数量（特定格式）
            print(f"diversity| distinct_answer_num: {distinct_answer_num} | all_answer_num: {all_answer_num} | distinct_answer_ratio: {distinct_answer_ratio:.2f} | best_answer_ratio: {best_answer_ratio:.2f} | correct_answer_number: {correct_answer_number} | best_is_correct: {best_is_correct} | extracted_answers: {extracted_answers} | majority_answer: {majority_answer} | ground_truth_answer: {answer}", flush=True)

            # 根据是否匹配多数投票结果分配奖励
            for i, ans in enumerate(extracted_answers):
                if ans == majority_answer:
                    rewards[start_idx + i] = 1.0  # 匹配多数投票结果
                # else:
                #     rewards[start_idx + i] = -1.0  # 不匹配多数投票结果
    
    return rewards

def reward_seq_entropy_rank(batch, responses, seq_log_probs_list, num_generations, device):
    """
    Compute reward based on sequence log probabilities ranking.
    """
    import numpy as np
    
    # 获取 ground truth 答案用于诊断
    ground_truth_cot = list(batch['answers'])[0]
    if "####" in ground_truth_cot:
        ground_truth_answer = extract_answer_gsm8k(ground_truth_cot)
    else:
        ground_truth_answer = parse_ground_truth(ground_truth_cot)[1]
    
    num_problems = len(responses) // num_generations
    rewards = torch.zeros(len(responses), device=device)
    
    # 统计
    top_correct_total = 0
    top_total = 0
    bottom_correct_total = 0
    bottom_total = 0
    
    for problem_idx in range(num_problems):
        start_idx = problem_idx * num_generations
        end_idx = start_idx + num_generations
        problem_seq_log_probs = seq_log_probs_list[start_idx:end_idx]
        problem_responses = responses[start_idx:end_idx]
        
        # 提取每个生成的答案和正确性
        results = []  # (log_prob, is_correct)
        for resp, log_prob in zip(problem_responses, problem_seq_log_probs):
            ans = extract_answer(resp)
            is_correct = (ans == ground_truth_answer)
            results.append((log_prob, is_correct))
        
        # 按 log_prob 排序
        results.sort(key=lambda x: x[0], reverse=True)
        
        # 前50% vs 后50%
        num_top = num_generations // 2
        if num_generations % 2 == 1:
            num_top = num_top + 1
        
        top_results = results[:num_top]
        bottom_results = results[num_top:]
        
        top_correct = sum(1 for _, correct in top_results if correct)
        bottom_correct = sum(1 for _, correct in bottom_results if correct)
        
        top_correct_total += top_correct
        top_total += len(top_results)
        bottom_correct_total += bottom_correct
        bottom_total += len(bottom_results)
        
        # 分配奖励
        sorted_indices = sorted(range(len(problem_seq_log_probs)), 
                               key=lambda i: problem_seq_log_probs[i], reverse=True)
        top_indices = sorted_indices[:num_top]
        
        for idx in top_indices:
            rewards[start_idx + idx] = 1.0
        # bottom 默认为 0
        
        # 简洁打印
        print(f"Problem {problem_idx}: Top acc={top_correct}/{len(top_results)} ({100*top_correct/len(top_results):.0f}%) | Bottom acc={bottom_correct}/{len(bottom_results)} ({100*bottom_correct/len(bottom_results):.0f}%)")
    
    # 总结
    top_acc = top_correct_total / top_total
    bottom_acc = bottom_correct_total / bottom_total
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: Top half accuracy = {top_acc*100:.1f}%")
    print(f"        Bottom half accuracy = {bottom_acc*100:.1f}%")
    print(f"        Gap = {(top_acc - bottom_acc)*100:.1f}%")
    
    if top_acc <= bottom_acc:
        print(f"⚠️  CRITICAL: Top accuracy <= Bottom accuracy! Ranking reward will hurt training.")
    elif top_acc - bottom_acc < 0.1:
        print(f"⚠️  WARNING: Gap is small (<10%), ranking signal is weak.")
    else:
        print(f"✅ Good: Top accuracy > Bottom accuracy by >10%, ranking should work.")
    print(f"{'='*50}\n")
    
    return rewards

def reward_seq_entropy(batch, responses, seq_log_probs_list, num_generations, device):
    num_problems = len(seq_log_probs_list) // num_generations
    rewards = torch.zeros(len(seq_log_probs_list), device=device)
    
    for i in range(num_problems):
        start = i * num_generations
        end = start + num_generations
        lp = torch.tensor(seq_log_probs_list[start:end], device=device)
        rewards[start:end] = lp  # 直接存原始log_prob，不做RLOO
        raw_log_probs = lp.mean().item()
        print(f"raw_log_prob mean: {raw_log_probs:.4f}")
    
    print("rewards: {}".format(rewards))
    return rewards

def load_gsm8k_dataset_and_reward_justgrpo(
    local_path: str = "gsm8k",
    batch_size: int = 1,
    split: str = 'train',
    num_workers: int = 4,
    seed: int = 112,
):
    """
    Load GSM8K dataset and return dataloader with reward function.
    
    Args:
        local_path: HuggingFace dataset path
        batch_size: Batch size per GPU
        split: Dataset split to use
        num_workers: Number of dataloader workers
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (dataloader, reward_function)
    """
    ds = load_dataset(local_path, "main", split=split)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    
    sampler = InfiniteSampler(
        ds, 
        rank=get_rank(), 
        num_replicas=get_world_size(),
    )
    
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_gsm8k,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, reward_gsm8k


def load_gsm8k_dataset_and_reward(
    local_path: str = "gsm8k",
    batch_size: int = 1,
    split: str = 'train',
    num_workers: int = 4,
    seed: int = 112,
    method: str = 'ttrl',
):
    """
    Load GSM8K dataset and return dataloader with reward function.
    
    Args:
        local_path: HuggingFace dataset path
        batch_size: Batch size per GPU
        split: Dataset split to use
        num_workers: Number of dataloader workers
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (dataloader, reward_function)
    """
    ds = load_dataset(local_path, "main", split=split)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    
    sampler = InfiniteSampler(
        ds, 
        rank=get_rank(), 
        num_replicas=get_world_size(),
    )
    
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_gsm8k,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    if method == 'ttrl':
        reward_fn = reward_ttrl
    elif method == 'seq_entropy':
        reward_fn = reward_seq_entropy
    return dataloader, reward_fn


def load_math500_dataset_and_reward(
    local_path: str = "HuggingFaceH4/MATH-500",
    batch_size: int = 1,
    split: str = 'test',
    num_workers: int = 4,
    seed: int = 112,
    method: str = 'ttrl',
    max_level: int = 3,
):
    """
    Load GSM8K dataset and return dataloader with reward function.
    
    Args:
        local_path: HuggingFace dataset path
        batch_size: Batch size per GPU
        split: Dataset split to use
        num_workers: Number of dataloader workers
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (dataloader, reward_function)
    """
    ds = load_dataset(local_path, "default", split=split)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    ds = ds.filter(lambda x: x['level'] <= max_level)  # 只训练难度1-3的题目
    print(f"Filtered MATH-500 dataset to max_level={max_level}, resulting in {len(ds)} samples.")
    sampler = InfiniteSampler(
        ds, 
        rank=get_rank(), 
        num_replicas=get_world_size(),
    )
    
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_math,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    if method == 'ttrl':
        reward_fn = reward_ttrl
    elif method == 'seq_entropy':
        reward_fn = reward_seq_entropy
    return dataloader, reward_fn

def load_aime2024_dataset_and_reward(
    local_path: str = "Maxwell-Jia/AIME_2024",
    batch_size: int = 1,
    split: str = 'train',
    num_workers: int = 4,
    seed: int = 112,
):
    """
    Load GSM8K dataset and return dataloader with reward function.
    
    Args:
        local_path: HuggingFace dataset path
        batch_size: Batch size per GPU
        split: Dataset split to use
        num_workers: Number of dataloader workers
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (dataloader, reward_function)
    """
    ds = load_dataset(local_path, "default", split=split)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    
    sampler = InfiniteSampler(
        ds, 
        rank=get_rank(), 
        num_replicas=get_world_size(),
    )
    
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_aime2024,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, reward_gsm8k_ttrl

def load_arc_dataset_and_reward(
    local_path: str = "arc",
    batch_size: int = 1,
    split: str = 'test',
    num_workers: int = 4,
    seed: int = 112,
    method: str = 'ttrl',
):
    """
    Load GSM8K dataset and return dataloader with reward function.
    
    Args:
        local_path: HuggingFace dataset path
        batch_size: Batch size per GPU
        split: Dataset split to use
        num_workers: Number of dataloader workers
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (dataloader, reward_function)
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    
    sampler = InfiniteSampler(
        ds, 
        rank=get_rank(), 
        num_replicas=get_world_size(),
    )
    
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_arc,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    if method == 'ttrl':
        reward_fn = reward_ttrl
    elif method == 'seq_entropy':
        reward_fn = reward_seq_entropy
    return dataloader, reward_fn
