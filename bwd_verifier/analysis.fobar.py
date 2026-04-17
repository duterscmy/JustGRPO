import json
import argparse
from collections import Counter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grader import math_equal
from utils.parser import parse_ground_truth


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_fobar_cases(dataset):
    """分析 voting 和 fobar 的对错关系"""
    
    voting_correct_fobar_wrong = []  # voting对，fobar错
    voting_wrong_fobar_correct = []  # voting错，fobar对
    both_correct = []
    both_wrong = []
    
    voting_correct_count = 0
    fobar_correct_count = 0
    
    for idx, sample in enumerate(dataset):
        rollouts = sample.get('rollouts', [])
        ground_truth = sample.get('answer', '')
        
        if not rollouts or not ground_truth:
            continue
        
        # 提取所有答案
        answers = [parse_ground_truth(r)[1] for r in rollouts]
        
        # Voting: 多数投票
        answer_counts = Counter(answers)
        voting_answer = answer_counts.most_common(1)[0][0]
        voting_correct = math_equal(voting_answer, ground_truth)
        
        # FOBAR: 从 backward_result 读取
        backward_results = sample.get('backward_result', [])
        if not backward_results:
            continue
        
        # 构建候选答案到 backward_score 的映射
        candidate_to_score = {}
        for br in backward_results:
            candidate = br.get('candidate', '')
            score = br.get('backward_score', 0)
            candidate_to_score[candidate] = score
        
        # 获取所有唯一答案
        unique_answers = list(set(answers))
        
        # 计算前向分数
        forward_scores = {ans: answers.count(ans)/len(answers) for ans in unique_answers}
        
        # 计算组合分数（几何平均）
        combined_scores = {}
        for ans in unique_answers:
            f = forward_scores.get(ans, 0)
            b = candidate_to_score.get(ans, 0)
            if b == 0.0:
                b = 0.01  # 避免乘积为0导致的几何平均问题
            combined_scores[ans] = (f * b) ** 0.5 if f > 0 and b > 0 else 0
        
        if combined_scores:
            fobar_answer = max(combined_scores, key=combined_scores.get)
        else:
            fobar_answer = voting_answer
        
        fobar_correct = math_equal(fobar_answer, ground_truth)
        
        # 统计
        if voting_correct:
            voting_correct_count += 1
        if fobar_correct:
            fobar_correct_count += 1
        
        # 分类记录
        case_info = {
            "sample_idx": idx,
            "question": sample.get('question', sample.get('prompt', ''))[:300],
            "ground_truth": ground_truth,
            "voting_answer": voting_answer,
            "fobar_answer": fobar_answer,
            "answers": answers,
            "answer_counts": dict(answer_counts),
            "forward_scores": forward_scores,
            "backward_scores": {k: candidate_to_score.get(k, 0) for k in unique_answers},
            "combined_scores": combined_scores
        }
        
        if voting_correct and not fobar_correct:
            voting_correct_fobar_wrong.append(case_info)
        elif not voting_correct and fobar_correct:
            voting_wrong_fobar_correct.append(case_info)
        elif voting_correct and fobar_correct:
            both_correct.append(case_info)
        else:
            both_wrong.append(case_info)
    
    return {
        "voting_correct_fobar_wrong": voting_correct_fobar_wrong,
        "voting_wrong_fobar_correct": voting_wrong_fobar_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "voting_correct_count": voting_correct_count,
        "fobar_correct_count": fobar_correct_count,
        "total_samples": len([s for s in dataset if s.get('rollouts') and s.get('answer')])
    }


def analyze_failure_reasons(cases, case_name):
    """分析失败原因"""
    reason_stats = {
        "correct_backward_score_0": 0,      # 正确答案 backward_score=0
        "wrong_backward_score_1": 0,        # 错误答案 backward_score=1
        "correct_backward_score_low": 0,    # 正确答案 backward_score < 0.3
        "wrong_backward_score_high": 0,     # 错误答案 backward_score > 0.7
        "forward_tie": 0,                   # 前向分数接近 (<0.1差异)
        "other": 0
    }
    
    for case in cases:
        correct_ans = case["ground_truth"]
        selected_ans = case["fobar_answer"] if "fobar" in case_name else case["voting_answer"]
        
        bwd_scores = case["backward_scores"]
        fwd_scores = case["forward_scores"]
        
        correct_bwd = bwd_scores.get(correct_ans, -1)
        selected_bwd = bwd_scores.get(selected_ans, -1)
        
        # 找出所有候选的前向分数
        fwd_values = list(fwd_scores.values())
        max_fwd = max(fwd_values)
        second_fwd = sorted(fwd_values, reverse=True)[1] if len(fwd_values) > 1 else 0
        
        if correct_bwd == 0:
            reason_stats["correct_backward_score_0"] += 1
        elif selected_bwd == 1:
            reason_stats["wrong_backward_score_1"] += 1
        elif correct_bwd < 0.3:
            reason_stats["correct_backward_score_low"] += 1
        elif selected_bwd > 0.7:
            reason_stats["wrong_backward_score_high"] += 1
        elif max_fwd - second_fwd < 0.1:
            reason_stats["forward_tie"] += 1
        else:
            reason_stats["other"] += 1
    
    return reason_stats


def print_analysis(results):
    print("=" * 80)
    print("FOBAR vs Voting Analysis")
    print("=" * 80)
    
    total = results["total_samples"]
    voting_correct = results["voting_correct_count"]
    fobar_correct = results["fobar_correct_count"]
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Voting correct: {voting_correct} ({voting_correct/total*100:.1f}%)")
    print(f"  FOBAR correct: {fobar_correct} ({fobar_correct/total*100:.1f}%)")
    print(f"  Improvement: {fobar_correct - voting_correct:+.1f}%")
    
    print(f"\n📈 Case Breakdown:")
    print(f"  ✅ Voting Correct → FOBAR Correct: {len(results['both_correct'])}")
    print(f"  ❌ Voting Correct → FOBAR Wrong:  {len(results['voting_correct_fobar_wrong'])}")
    print(f"  ✅ Voting Wrong  → FOBAR Correct: {len(results['voting_wrong_fobar_correct'])}")
    print(f"  ❌ Voting Wrong  → FOBAR Wrong:   {len(results['both_wrong'])}")
    
    # 净收益
    net_gain = len(results['voting_wrong_fobar_correct']) - len(results['voting_correct_fobar_wrong'])
    print(f"\n📉 Net Gain: {net_gain:+d} ({net_gain/total*100:+.1f}%)")
    
    # 分析 voting correct → fobar wrong 的原因
    if results['voting_correct_fobar_wrong']:
        print("\n" + "=" * 80)
        print("🔍 Why Voting Correct → FOBAR Wrong?")
        print("=" * 80)
        reasons = analyze_failure_reasons(results['voting_correct_fobar_wrong'], "fobar")
        for reason, count in reasons.items():
            if count > 0:
                print(f"  {reason}: {count}")
    
    # 分析 voting wrong → fobar correct 的原因
    if results['voting_wrong_fobar_correct']:
        print("\n" + "=" * 80)
        print("🎯 Why Voting Wrong → FOBAR Correct? (Success Cases)")
        print("=" * 80)
        reasons = analyze_failure_reasons(results['voting_wrong_fobar_correct'], "voting")
        for reason, count in reasons.items():
            if count > 0:
                print(f"  {reason}: {count}")
    
    # 打印详细样例
    print("\n" + "=" * 80)
    print("📋 Detailed Examples")
    print("=" * 80)
    
    # Voting Correct → FOBAR Wrong (前3个)
    if results['voting_correct_fobar_wrong']:
        print("\n❌ Voting Correct → FOBAR Wrong (first 3):")
        print("-" * 60)
        for i, case in enumerate(results['voting_correct_fobar_wrong'][:3]):
            print(f"\n[{i+1}] Sample {case['sample_idx']}")
            print(f"  Question: {case['question'][:150]}...")
            print(f"  Ground Truth: {case['ground_truth']}")
            print(f"  Voting Answer: {case['voting_answer']} ✓")
            print(f"  FOBAR Answer: {case['fobar_answer']} ✗")
            print(f"  Forward scores: {case['forward_scores']}")
            print(f"  Backward scores: {case['backward_scores']}")
            print(f"  Combined scores: {case['combined_scores']}")
    
    # Voting Wrong → FOBAR Correct (前3个)
    if results['voting_wrong_fobar_correct']:
        print("\n✅ Voting Wrong → FOBAR Correct (first 3):")
        print("-" * 60)
        for i, case in enumerate(results['voting_wrong_fobar_correct'][:3]):
            print(f"\n[{i+1}] Sample {case['sample_idx']}")
            print(f"  Question: {case['question'][:150]}...")
            print(f"  Ground Truth: {case['ground_truth']}")
            print(f"  Voting Answer: {case['voting_answer']} ✗")
            print(f"  FOBAR Answer: {case['fobar_answer']} ✓")
            print(f"  Forward scores: {case['forward_scores']}")
            print(f"  Backward scores: {case['backward_scores']}")
            print(f"  Combined scores: {case['combined_scores']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, 
                        default='math500_results.add_records.add_digits_bwd.json',
                        help='Input JSON file with backward_result')
    parser.add_argument('--output', '-o', type=str, default='fobar_analysis.json',
                        help='Output file for detailed results')
    args = parser.parse_args()
    
    dataset = load_data(args.input_file)
    print(f"Loaded {len(dataset)} samples")
    
    results = analyze_fobar_cases(dataset)
    print_analysis(results)
    
    # 保存详细结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()