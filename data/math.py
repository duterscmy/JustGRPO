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
from utils.parser import extract_answer


def collate_fn_gsm8k(batch):
    """Collate function for GSM8K dataset."""
    problems = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    return {"problems": problems, "answers": answers}

def collate_fn_math500(batch):
    """Collate function for GSM8K dataset."""
    problems = [item['problem'] for item in batch]
    answers = [item['solution'] for item in batch]
    return {"problems": problems, "answers": answers}

def collate_fn_aime2024(batch):
    """Collate function for GSM8K dataset."""
    problems = [item['Problem'] for item in batch]
    answers = [item['Solution'] for item in batch]
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


def reward_gsm8k_ttrl(batch, responses, num_generations, device):
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
    
    # 将 responses 按问题分组
    answer = list(batch['answers'])[0]
    print("correct answer: {}".format(answer))
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
            print(f"diversity| distinct_answer_num: {distinct_answer_num} | all_answer_num: {all_answer_num} | distinct_answer_ratio: {distinct_answer_ratio:.2f} | best_answer_ratio: {best_answer_ratio:.2f} | correct_answer_number: {correct_answer_number} | best_is_correct: {best_is_correct}", flush=True)

            # 根据是否匹配多数投票结果分配奖励
            for i, ans in enumerate(extracted_answers):
                if ans == majority_answer:
                    rewards[start_idx + i] = 1.0  # 匹配多数投票结果
                # else:
                #     rewards[start_idx + i] = -1.0  # 不匹配多数投票结果
    
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
    
    return dataloader, reward_gsm8k_ttrl


def load_math500_dataset_and_reward(
    local_path: str = "HuggingFaceH4/MATH-500",
    batch_size: int = 1,
    split: str = 'test',
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
        collate_fn=collate_fn_math500,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, reward_gsm8k_ttrl

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
