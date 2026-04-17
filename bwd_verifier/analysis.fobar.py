import json
import argparse
from collections import Counter
from utils.grader import math_equal
from utils.parser import parse_ground_truth


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_fobar_failures(dataset):
    """分析 voting 正确但 fobar 错误的样本"""
    
    failures = []
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
            combined_scores[ans] = (f * b) ** 0.5 if f > 0 and b > 0 else 0
        
        if combined_scores:
            fobar_answer = max(combined_scores, key=combined_scores.get)
        else:
            fobar_answer = voting_answer
        
        fobar_correct = math_equal(fobar_answer, ground_truth)
        
        if voting_correct:
            voting_correct_count += 1
            if not fobar_correct:
                # Voting 正确但 FOBAR 错误
                failures.append({
                    "sample_idx": idx,
                    "question": sample.get('question', sample.get('prompt', ''))[:200],
                    "ground_truth": ground_truth,
                    "voting_answer": voting_answer,
                    "fobar_answer": fobar_answer,
                    "answers": answers,
                    "answer_counts": dict(answer_counts),
                    "forward_scores": forward_scores,
                    "backward_scores": {k: candidate_to_score.get(k, 0) for k in unique_answers},
                    "combined_scores": combined_scores
                })
        
        if fobar_correct:
            fobar_correct_count += 1
    
    return failures, voting_correct_count, fobar_correct_count


def print_analysis(failures, voting_correct, fobar_correct):
    print("=" * 80)
    print("FOBAR Failure Analysis")
    print("=" * 80)
    print(f"Voting correct samples: {voting_correct}")
    print(f"FOBAR correct samples: {fobar_correct}")
    print(f"Voting correct but FOBAR wrong: {len(failures)}")
    print(f"FOBAR degradation: {len(failures)}/{voting_correct} = {len(failures)/voting_correct*100:.1f}%")
    print()
    
    # 分类失败原因
    reason_stats = {
        "backward_score_0_for_correct": 0,
        "backward_score_1_for_wrong": 0,
        "multiple_candidates_close": 0,
        "no_backward_info": 0
    }
    
    for f in failures:
        correct_ans = f["ground_truth"]
        voting_ans = f["voting_answer"]
        fobar_ans = f["fobar_answer"]
        
        bwd_scores = f["backward_scores"]
        fwd_scores = f["forward_scores"]
        
        # 检查正确候选的 backward_score
        correct_bwd = bwd_scores.get(correct_ans, -1)
        fobar_bwd = bwd_scores.get(fobar_ans, -1)
        
        if correct_bwd == 0:
            reason_stats["backward_score_0_for_correct"] += 1
        elif fobar_bwd == 1:
            reason_stats["backward_score_1_for_wrong"] += 1
        elif abs(fwd_scores.get(correct_ans, 0) - fwd_scores.get(fobar_ans, 0)) < 0.1:
            reason_stats["multiple_candidates_close"] += 1
        else:
            reason_stats["no_backward_info"] += 1
    
    print("Failure Reasons:")
    for reason, count in reason_stats.items():
        print(f"  {reason}: {count}")
    print()
    
    # 打印详细样例
    print("Detailed Failure Examples (first 10):")
    print("-" * 80)
    for i, f in enumerate(failures[:10]):
        print(f"\n[Sample {f['sample_idx']}]")
        print(f"Question: {f['question']}...")
        print(f"Ground Truth: {f['ground_truth']}")
        print(f"Voting Answer: {f['voting_answer']} (correct)")
        print(f"FOBAR Answer: {f['fobar_answer']} (wrong)")
        print(f"All answers: {f['answers']}")
        print(f"Answer counts: {f['answer_counts']}")
        print(f"Forward scores: {f['forward_scores']}")
        print(f"Backward scores: {f['backward_scores']}")
        print(f"Combined scores: {f['combined_scores']}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, 
                        default='math500_results.add_records.add_digits_bwd.json',
                        help='Input JSON file with backward_result')
    parser.add_argument('--output', '-o', type=str, default='fobar_failures.json',
                        help='Output file for failures')
    args = parser.parse_args()
    
    dataset = load_data(args.input_file)
    print(f"Loaded {len(dataset)} samples")
    
    failures, voting_correct, fobar_correct = analyze_fobar_failures(dataset)
    print_analysis(failures, voting_correct, fobar_correct)
    
    # 保存失败样例
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    print(f"\nFailures saved to {args.output}")


if __name__ == "__main__":
    main()