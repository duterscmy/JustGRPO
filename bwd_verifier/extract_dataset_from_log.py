#!/usr/bin/env python3
"""
从日志文件中提取rollout数据并生成JSON数据集
"""

import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_log_file(log_file_path):
    """
    解析日志文件，提取每个问题的rollout和ground truth
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割每个问题块（从user开始到下一个统计信息结束）
    # 匹配每个问题的完整内容
    pattern = r'(user\n.*?==========MAJORITY:.*?===========\n)'
    
    # 使用finditer来获取所有匹配
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    print(f"Found {len(matches)} questions")
    
    dataset = []
    rollouts_list = []
    for match in tqdm(matches, desc="Processing questions"):
        block = match.group(1)
        
        # 提取8个rollout
        # 每个rollout从 "user" 开始到下一个 "user" 或 "==========MAJORITY" 结束
        rollout_pattern = r'user\n(.*?)assistant\n(.*?)\n=================='
        rollouts_raw = re.findall(rollout_pattern, block, re.DOTALL)
        
        # 整理rollout格式
        rollouts = []
        for user_msg, assistant_msg in rollouts_raw:
            # 清理文本，去除多余空白
            user_msg = user_msg.strip()
            assistant_msg = assistant_msg.strip()
            # assistant_msg = assistant_msg.rsplit('\n',1)[0].strip()  # 去掉最后一行的统计信息
            # 提取assistant的答案（最后一行通常是数字）
            lines = assistant_msg.strip().split('\n')
            answer = lines[-1].strip() if lines else ""
            
            rollout = {
                "user": user_msg,
                "assistant": assistant_msg,
                "extracted_answer": answer
            }
            rollouts.append(rollout)
        rollouts_list.append(rollouts)  # 用于后续统计分析


        
    stats_line_list = []
    stats_list = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "diversity" in line:
                stats_line_list.append(line.strip())
    print(len(rollouts_list),len(stats_line_list))
    assert len(rollouts_list) == len(stats_line_list)

    for stats_line, rollouts in zip(stats_line_list, rollouts_list):
        ground_truth = None
        if stats_line:
            gt_match = re.search(r'ground_truth_answer:\s*(\d+)', stats_line)
            ground_truth = int(gt_match.group(1)) if gt_match else None
        
        # 提取其他统计信息（可选）
        majority = None
        distinct_answer_num = None
        best_answer_ratio = None
        best_is_correct = None
        extracted_answers = []
        
        if stats_line:
            majority_match = re.search(r'MAJORITY:\s*(\d+)', stats_line)
            majority = int(majority_match.group(1)) if majority_match else None
            
            distinct_match = re.search(r'distinct_answer_num:\s*(\d+)', stats_line)
            distinct_answer_num = int(distinct_match.group(1)) if distinct_match else None
            
            ratio_match = re.search(r'best_answer_ratio:\s*([\d.]+)', stats_line)
            best_answer_ratio = float(ratio_match.group(1)) if ratio_match else None
            
            correct_match = re.search(r'best_is_correct:\s*(\d+)', stats_line)
            best_is_correct = int(correct_match.group(1)) if correct_match else None
            
            answers_match = re.search(r'extracted_answers:\s*\[(.*?)\]', stats_line)
            if answers_match:
                answers_str = answers_match.group(1)
                extracted_answers = [ans.strip().strip("'\"") for ans in answers_str.split(',')]
        
        # 构建数据条目
        new_rollouts = []
        for rollout in rollouts:
            print(rollout["extracted_answer"], str(ground_truth))
            print(rollout["extracted_answer"] == str(ground_truth))
            if rollout["extracted_answer"] == str(ground_truth):
                rollout["is_correct"] = 1
                print("correct!")
            else:
                rollout["is_correct"] = 0
            new_rollouts.append(rollout)
        rollouts = new_rollouts
        data_item = {
            "rollouts": rollouts,
            "ground_truth_answer": ground_truth,
            "statistics": {
                "majority_answer": majority,
                "distinct_answer_num": distinct_answer_num,
                "best_answer_ratio": best_answer_ratio,
                "best_is_correct": best_is_correct,
                "extracted_answers": extracted_answers
            }
        }
        
        dataset.append(data_item)
    
    return dataset


def save_dataset(dataset, output_path):
    """
    保存数据集为JSON文件
    """
    output_file = Path(output_path)
    
    # 保存为JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Dataset saved to {output_path}")
    print(f"  Total questions: {len(dataset)}")
    
    # 统计信息
    total_rollouts = sum(len(item["rollouts"]) for item in dataset)
    print(f"  Total rollouts: {total_rollouts}")
    
    # 计算正确率
    correct_count = 0
    for item in dataset:
        stats = item.get("statistics", {})
        if stats.get("best_is_correct") == 1:
            correct_count += 1
    
    if dataset:
        print(f"  Best answer correct rate: {correct_count}/{len(dataset)} = {correct_count/len(dataset)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Extract rollout data from log file")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("output_file", help="Path to save the JSON dataset")
    parser.add_argument("max_samples", type=int, help="Maximum number of samples to extract")
    
    args = parser.parse_args()
    
    if not Path(args.log_file).exists():
        print(f"Error: Log file '{args.log_file}' not found")
        return 1
    
    print(f"Parsing log file: {args.log_file}")
    
    dataset = parse_log_file(args.log_file)
    dataset = dataset[:args.max_samples]  # 限制最大样本数
    if dataset:
        save_dataset(dataset, args.output_file)
        return 0
    else:
        print("No data extracted")
        return 1


if __name__ == "__main__":
    main()