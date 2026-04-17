import json
import argparse
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import numpy as np
import torch
import re
import sys, os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grader import math_equal
from utils.parser import extract_answer, parse_ground_truth


def normalize_answer_with_math_equal(answer: str, ground_truth: str = None) -> str:
    """
    使用 math_equal 对答案进行归一化
    对于数学等价的答案（如 4000 和 4,000），返回统一表示
    """
    if not answer:
        return answer
    
    # 如果有 ground_truth，尝试找到等价的标准形式
    if ground_truth is not None:
        if math_equal(answer, ground_truth):
            return ground_truth
    
    # 通用归一化：移除逗号、空格等
    normalized = answer.strip()
    # 移除数字中的逗号（如 "4,000" -> "4000"）
    normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
    # 移除首尾空格
    normalized = normalized.strip()
    
    return normalized


def normalize_answers_in_votes(votes: Dict[str, int]) -> Dict[str, int]:
    """
    对投票字典中的答案进行归一化，合并数学等价的答案
    使用 math_equal 进行两两比较（由于答案数量通常很小，O(n^2) 可接受）
    """
    if not votes:
        return votes
    
    answers = list(votes.keys())
    n = len(answers)
    
    # 找出等价类
    merged = {}
    used = set()
    
    for i in range(n):
        if i in used:
            continue
        ans_i = answers[i]
        # 找到与 ans_i 等价的所有答案
        equiv_class = [ans_i]
        for j in range(i + 1, n):
            if j in used:
                continue
            ans_j = answers[j]
            if math_equal(ans_i, ans_j):
                equiv_class.append(ans_j)
                used.add(j)
        # 选择代表元（出现次数最多的，或第一个）
        representative = max(equiv_class, key=lambda x: votes.get(x, 0))
        # 合并票数
        total_votes = sum(votes.get(ans, 0) for ans in equiv_class)
        merged[representative] = total_votes
        used.add(i)
    
    return merged


class AnswerSelector:
    """答案选择器，支持多种策略"""
    
    def __init__(self, strategy='first', combine_method='geometric'):
        """
        Args:
            strategy: 选择策略
                - 'first': 直接选择第一个rollout
                - 'majority': 多数投票
                - 'fobar': 前向+后向验证
            combine_method: 组合方式，'geometric' 或 'arithmetic'
        """
        self.strategy = strategy
        self.combine_method = combine_method
        self._cache = {}
        self._parsed_cache = {}
    
    def _get_parsed_answer(self, rollout: str) -> Tuple[str, str]:
        """获取解析后的答案（带缓存）"""
        if rollout not in self._parsed_cache:
            self._parsed_cache[rollout] = parse_ground_truth(rollout)
        return self._parsed_cache[rollout]

    def select_answer(self, sample: Dict) -> Tuple[str, Dict]:
        """根据策略选择最佳答案"""
        cache_key = (
            sample.get('question', sample.get('prompt', '')),
            self.strategy,
            self.combine_method
        )
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        rollouts = sample['rollouts']
        rollouts_records = sample.get('rollouts_records', [])
        
        if self.strategy == 'first':
            result = self._select_first(rollouts)
        elif self.strategy == 'majority':
            result = self._select_majority(rollouts)
        elif self.strategy == 'highest_confidence':
            result = self._select_highest_confidence(rollouts, rollouts_records)
        elif self.strategy == 'weighted_confidence':
            result = self._select_weighted_confidence(rollouts, rollouts_records)
        elif self.strategy == 'confidence_threshold':
            threshold = sample.get('confidence_threshold', 0.9)
            result = self._select_confidence_threshold(rollouts, rollouts_records, threshold)
        elif self.strategy == 'fobar':
            result = self._select_fobar(sample)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self._cache[cache_key] = result
        return result
    
    def _select_first(self, rollouts: List[str]) -> Tuple[str, Dict]:
        """直接选择第一个rollout的答案"""
        first_answer = self._get_parsed_answer(rollouts[0])[1]
        
        return first_answer, {
            "strategy": "first",
            "selected_index": 0,
        }
    
    def _select_majority(self, rollouts: List[str]) -> Tuple[str, Dict]:
        """多数投票选择答案（带数学等价归一化）"""
        answers = [self._get_parsed_answer(r)[1] for r in rollouts]
        
        # 使用 math_equal 进行归一化投票
        answer_counts = Counter(answers)
        normalized_counts = normalize_answers_in_votes(dict(answer_counts))
        
        most_common = max(normalized_counts.items(), key=lambda x: x[1])
        selected_answer = most_common[0]
        max_count = most_common[1]
        
        return selected_answer, {
            "strategy": "majority",
            "answer_counts_raw": dict(answer_counts),
            "answer_counts_normalized": normalized_counts,
            "selected_count": max_count,
            "total_rollouts": len(rollouts),
            "all_answers_raw": answers
        }
    
    def _select_fobar(self, sample: Dict) -> Tuple[str, Dict]:
        """
        使用FOBAR策略选择答案
        从 backward_result 字段读取后向分数
        """
        rollouts = sample['rollouts']
        
        # 1. 提取所有答案（使用 math_equal 归一化）
        answers_raw = [self._get_parsed_answer(r)[1] for r in rollouts]
        
        # 使用 math_equal 找到每个答案的等价代表元
        # 首先收集唯一答案
        unique_answers_raw = list(set(answers_raw))
        # 建立等价映射
        answer_to_representative = {}
        representatives = []
        for ans in unique_answers_raw:
            found = False
            for rep in representatives:
                if math_equal(ans, rep):
                    answer_to_representative[ans] = rep
                    found = True
                    break
            if not found:
                representatives.append(ans)
                answer_to_representative[ans] = ans
        
        # 归一化后的答案列表
        answers_normalized = [answer_to_representative[ans] for ans in answers_raw]
        unique_answers = list(set(answers_normalized))
        
        # 2. 计算前向分数（频率）- 基于归一化后的答案
        answer_counts = Counter(answers_normalized)
        forward_scores = {ans: count/len(rollouts) for ans, count in answer_counts.items()}
        
        # 3. 从 backward_result 读取后向分数
        backward_result = sample.get('backward_result', [])
        backward_scores_raw = {}
        for br in backward_result:
            candidate = br.get('candidate', '')
            # 找到该候选答案对应的代表元
            rep = answer_to_representative.get(candidate, candidate)
            backward_scores_raw[rep] = br.get('backward_score', 0.0)
        
        # 如果某些代表元没有 backward_score，给默认值 0
        for ans in unique_answers:
            if ans not in backward_scores_raw:
                backward_scores_raw[ans] = 0.0
        
        # 4. 组合分数
        combined_scores = {}
        for ans in unique_answers:
            f = forward_scores.get(ans, 0.0)
            b = backward_scores_raw.get(ans, 0.0)
            if b == 0.0:
                b = 0.01
            if self.combine_method == 'geometric':
                # 几何平均: sqrt(f * b)
                combined_scores[ans] = np.sqrt(f * b) if f > 0 and b > 0 else 0.0
            elif self.combine_method == 'arithmetic':
                # 算术平均: (f + b) / 2
                combined_scores[ans] = (f + b) / 2
            else:
                raise ValueError(f"Unknown combine_method: {self.combine_method}")
        
        # 5. 选择分数最高的答案
        if combined_scores:
            selected_answer = max(combined_scores, key=combined_scores.get)
        else:
            selected_answer = unique_answers[0] if unique_answers else ""
        
        return selected_answer, {
            "strategy": "fobar",
            "combine_method": self.combine_method,
            "forward_scores": forward_scores,
            "backward_scores": backward_scores_raw,
            "combined_scores": combined_scores,
            "all_answers_raw": answers_raw,
            "all_answers_normalized": answers_normalized,
            "answer_mapping": answer_to_representative
        }
    
    def _extract_numbers(self, text: str) -> List[Tuple[str, Tuple[int, int], str]]:
        """从文本中提取数字（保留用于其他策略）"""
        pattern = r'-?\d+\.?\d*'
        numbers = []
        for match in re.finditer(pattern, text):
            num_str = match.group()
            if num_str and num_str not in ['-', '.']:
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                context = text[start:end]
                numbers.append((num_str, (match.start(), match.end()), context))
        return numbers

    def _select_highest_confidence(self, rollouts: List[str], rollouts_records: List[List[Dict]]) -> Tuple[str, Dict]:
        """
        选择置信度最高的rollout的答案
        """
        if not rollouts_records or len(rollouts_records) != len(rollouts):
            print("Warning: No confidence records found, falling back to first rollout")
            return self._select_first(rollouts)
        
        confidence_scores = []
        
        for idx, records in enumerate(rollouts_records):
            token_confidences = [record.get('confidence', 0.0) for record in records]
            
            if token_confidences:
                score = sum(token_confidences) / len(token_confidences)
                confidence_scores.append({
                    "rollout_idx": idx,
                    "score": score,
                    "avg_confidence": score,
                    "num_tokens": len(token_confidences),
                })
            else:
                confidence_scores.append({
                    "rollout_idx": idx,
                    "score": 0.0,
                    "avg_confidence": 0.0,
                    "num_tokens": 0,
                })
        
        best_rollout = max(confidence_scores, key=lambda x: x['score'])
        best_idx = best_rollout['rollout_idx']
        selected_answer = self._get_parsed_answer(rollouts[best_idx])[1]
        
        return selected_answer, {
            "strategy": "highest_confidence",
            "selected_index": best_idx,
            "selected_confidence": best_rollout['score'],
            "selected_confidence_details": {
                "avg": best_rollout['avg_confidence'],
                "num_tokens": best_rollout['num_tokens']
            },
        }

    def _select_weighted_confidence(self, rollouts: List[str], rollouts_records: List[List[Dict]]) -> Tuple[str, Dict]:
        """
        基于置信度加权投票选择答案
        """
        if not rollouts_records or len(rollouts_records) != len(rollouts):
            print("Warning: No confidence records found, falling back to majority")
            return self._select_majority(rollouts)
        
        answer_confidence = {}
        answer_votes = {}
        
        for idx, records in enumerate(rollouts_records):
            answer = self._get_parsed_answer(rollouts[idx])[1]
            token_confidences = [record.get('confidence', 0.0) for record in records]
            
            if token_confidences:
                weight = sum(token_confidences) / len(token_confidences)
            else:
                weight = 0.0
            
            if answer not in answer_confidence:
                answer_confidence[answer] = 0.0
                answer_votes[answer] = 0
            answer_confidence[answer] += weight
            answer_votes[answer] += 1
        
        selected_answer = max(answer_confidence, key=answer_confidence.get)
        
        all_answers = [self._get_parsed_answer(r)[1] for r in rollouts]
        answer_counts = Counter(all_answers)
        forward_scores = {ans: count/len(rollouts) for ans, count in answer_counts.items()}
        
        return selected_answer, {
            "strategy": "weighted_confidence",
            "selected_answer": selected_answer,
            "selected_confidence": answer_confidence[selected_answer],
            "answer_confidence_scores": answer_confidence,
            "answer_votes": answer_votes,
            "forward_scores": forward_scores,
        }

    def _select_confidence_threshold(self, rollouts: List[str], rollouts_records: List[List[Dict]], 
                                    threshold: float = 0.9) -> Tuple[str, Dict]:
        """
        基于置信度阈值选择
        """
        if not rollouts_records or len(rollouts_records) != len(rollouts):
            print("Warning: No confidence records found, falling back to majority")
            return self._select_majority(rollouts)
        
        high_confidence_answers = []
        rollout_filter_details = []
        
        for idx, records in enumerate(rollouts_records):
            token_confidences = [record.get('confidence', 0.0) for record in records]
            
            if token_confidences:
                avg_confidence = sum(token_confidences) / len(token_confidences)
            else:
                avg_confidence = 0.0
            
            answer = self._get_parsed_answer(rollouts[idx])[1]
            
            rollout_filter_details.append({
                "rollout_idx": idx,
                "answer": answer,
                "avg_confidence": avg_confidence,
                "passed_threshold": avg_confidence >= threshold
            })
            
            if avg_confidence >= threshold:
                high_confidence_answers.append(answer)
        
        if not high_confidence_answers:
            print(f"Warning: No rollout meets confidence threshold {threshold}, falling back to majority")
            return self._select_majority(rollouts)
        
        answer_counts = Counter(high_confidence_answers)
        most_common = answer_counts.most_common(1)[0]
        selected_answer = most_common[0]
        max_count = most_common[1]
        
        return selected_answer, {
            "strategy": "confidence_threshold",
            "threshold": threshold,
            "selected_answer": selected_answer,
            "selected_count": max_count,
            "total_high_confidence_rollouts": len(high_confidence_answers),
            "answer_counts": dict(answer_counts),
            "rollout_filter_details": rollout_filter_details,
        }


def evaluate_single_sample(args_tuple):
    """评估单个样本（用于多进程）"""
    sample, strategy, ground_truth_key, combine_method, sample_idx = args_tuple
    
    selector = AnswerSelector(strategy=strategy, combine_method=combine_method)
    
    try:
        selected_answer, info = selector.select_answer(sample)
    except Exception as e:
        print(f"❌ Error in sample {sample_idx}: {e}")
        traceback.print_exc()
        selected_answer = ""
        info = {"error": str(e)}
    
    ground_truth = sample.get(ground_truth_key, '')
    is_correct = math_equal(selected_answer, ground_truth)
    
    return {
        "sample_idx": sample_idx,
        "strategy": strategy,
        "selected_answer": selected_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "details": info
    }


class ParallelEvaluator:
    """并行评估器"""
    
    def __init__(self, strategies: List[str] = ['first', 'majority'], 
                 combine_method: str = 'geometric',
                 num_workers: int = None,
                 ground_truth_key: str = 'answer'):
        self.strategies = strategies
        self.combine_method = combine_method
        self.num_workers = num_workers or cpu_count()
        self.ground_truth_key = ground_truth_key
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """并行评估整个数据集"""
        results = {}
        
        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy}")
            if strategy == 'fobar':
                print(f"Combine method: {self.combine_method}")
            print(f"Using {self.num_workers} workers")
            print(f"Dataset size: {len(dataset)}")
            print(f"{'='*60}")
            
            args_list = [
                (sample, strategy, self.ground_truth_key, self.combine_method, idx)
                for idx, sample in enumerate(dataset)
            ]
            
            start_time = time.time()
            
            with Pool(processes=self.num_workers) as pool:
                unordered_results = []
                for result in tqdm(
                    pool.imap_unordered(evaluate_single_sample, args_list),
                    total=len(dataset),
                    desc=f"Processing {strategy}",
                    unit="sample",
                    ncols=80
                ):
                    unordered_results.append(result)
            
            elapsed = time.time() - start_time
            speed = len(dataset) / elapsed
            print(f"\n⏱️  Strategy {strategy} completed in {elapsed:.1f}s ({speed:.2f} samples/sec)")
            
            unordered_results.sort(key=lambda x: x['sample_idx'])
            for result in unordered_results:
                del result['sample_idx']
            
            sample_results = unordered_results
            correct_count = sum(1 for r in sample_results if r['is_correct'])
            accuracy = correct_count / len(dataset) if dataset else 0
            
            results[strategy] = {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": len(dataset),
                "details": sample_results
            }
            
            print(f"\n{strategy.upper()} Accuracy: {correct_count}/{len(dataset)} = {accuracy*100:.2f}%")
        
        return results


class SequentialEvaluator:
    """顺序评估器"""
    
    def __init__(self, strategies: List[str] = ['first', 'majority'], 
                 combine_method: str = 'geometric',
                 ground_truth_key: str = 'answer'):
        self.strategies = strategies
        self.combine_method = combine_method
        self.ground_truth_key = ground_truth_key
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """顺序评估整个数据集"""
        results = {}
        
        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy}")
            if strategy == 'fobar':
                print(f"Combine method: {self.combine_method}")
            print(f"{'='*60}")
            
            start_time = time.time()
            correct_count = 0
            sample_results = []
            selector = AnswerSelector(strategy=strategy, combine_method=self.combine_method)
            
            for i, sample in enumerate(tqdm(dataset, desc=f"Processing {strategy}", unit="sample")):
                selected_answer, info = selector.select_answer(sample)
                ground_truth = sample.get(self.ground_truth_key, '')
                is_correct = math_equal(selected_answer, ground_truth)
                
                sample_results.append({
                    "strategy": strategy,
                    "selected_answer": selected_answer,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct,
                    "details": info
                })
                
                if is_correct:
                    correct_count += 1
            
            elapsed = time.time() - start_time
            speed = len(dataset) / elapsed
            print(f"\n⏱️  Strategy {strategy} completed in {elapsed:.1f}s ({speed:.2f} samples/sec)")
            
            accuracy = correct_count / len(dataset) if dataset else 0
            results[strategy] = {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": len(dataset),
                "details": sample_results
            }
            
            print(f"\n{strategy.upper()} Accuracy: {correct_count}/{len(dataset)} = {accuracy*100:.2f}%")
        
        return results


def load_dataset(filepath: str) -> List[Dict]:
    """加载数据集"""
    print(f"Loading dataset from {filepath}")
    start_time = time.time()
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    elapsed = time.time() - start_time
    print(f"Loaded {len(dataset)} samples in {elapsed:.2f}s")
    
    # 检查数据格式
    if dataset:
        sample0 = dataset[0]
        print(f"\n📋 Sample structure:")
        print(f"  Keys: {list(sample0.keys())}")
        if 'rollouts' in sample0:
            print(f"  Number of rollouts: {len(sample0['rollouts'])}")
        if 'backward_result' in sample0:
            print(f"  Has backward_result: True")
            br = sample0['backward_result']
            if br:
                print(f"    First candidate: {br[0].get('candidate', 'N/A')} -> score: {br[0].get('backward_score', 'N/A')}")
    
    return dataset


def save_results(results: Dict, output_path: str):
    """保存评估结果"""
    summary = {
        "strategies": {},
        "total_samples": 0,
        "combine_method": results.get('combine_method', 'geometric')
    }
    
    for strategy, data in results.items():
        if strategy == 'combine_method':
            continue
        summary["strategies"][strategy] = {
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"]
        }
        if not summary["total_samples"]:
            summary["total_samples"] = data["total"]
    
    output_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for strategy, data in summary["strategies"].items():
        print(f"{strategy.upper()}: {data['correct']}/{data['total']} = {data['accuracy']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate different answer selection strategies')
    parser.add_argument('input_file', type=str, help='Input JSON file path (with backward_result)')
    parser.add_argument('output_file', type=str, help='Output JSON file path')
    parser.add_argument('--strategies', '-s', type=str, nargs='+', 
                        default=['first', 'majority', 'fobar'],
                        choices=['first', 'majority', 'fobar', 'highest_confidence', 
                                'weighted_confidence', 'confidence_threshold'],
                        help='Strategies to evaluate')
    parser.add_argument('--combine_method', '-c', type=str, default='geometric',
                        choices=['geometric', 'arithmetic'],
                        help='Combine method for FOBAR (geometric or arithmetic)')
    parser.add_argument('--num_workers', '-n', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential evaluation (for debugging)')
    
    args = parser.parse_args()
    
    # 加载数据集（需要包含 backward_result）
    dataset = load_dataset(args.input_file)
    
    # 检查是否有 backward_result
    has_backward = any('backward_result' in s and s['backward_result'] for s in dataset)
    if 'fobar' in args.strategies and not has_backward:
        print("\n⚠️  Warning: 'fobar' strategy requires backward_result in dataset!")
        print("   Please run backward verification script first.")
        if 'fobar' in args.strategies:
            args.strategies.remove('fobar')
            print("   Removed fobar from strategies.")
    
    # 选择评估器
    if args.sequential:
        print("\n📋 Using sequential evaluation...")
        evaluator = SequentialEvaluator(
            strategies=args.strategies,
            combine_method=args.combine_method
        )
    else:
        print(f"\n✅ Using parallel evaluation with {args.num_workers or cpu_count()} workers...")
        evaluator = ParallelEvaluator(
            strategies=args.strategies,
            combine_method=args.combine_method,
            num_workers=args.num_workers
        )
    
    # 评估
    results = evaluator.evaluate_dataset(dataset)
    results['combine_method'] = args.combine_method
    
    # 保存结果
    save_results(results, args.output_file)


if __name__ == "__main__":
    main()