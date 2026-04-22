import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
import re
import sys, os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import traceback
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grader import math_equal
from utils.parser import extract_answer, parse_ground_truth

import signal

def timeout_handler(signum, frame):
    raise TimeoutError("math_equal timeout")

def safe_math_equal(a, b, timeout=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = math_equal(a, b)
        signal.alarm(0)
        return result
    except TimeoutError:
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────
# 归一化工具
# ─────────────────────────────────────────────

def normalize_answers_in_votes(votes: Dict[str, int]) -> Dict[str, int]:
    answers = list(votes.keys())
    merged = {}
    used = set()
    for i in range(len(answers)):
        if i in used:
            continue
        equiv = [answers[i]]
        for j in range(i + 1, len(answers)):
            if j in used:
                continue
            if safe_math_equal(answers[i], answers[j]):
                equiv.append(answers[j])
                used.add(j)
        rep = max(equiv, key=lambda x: votes.get(x, 0))
        merged[rep] = sum(votes.get(a, 0) for a in equiv)
        used.add(i)
    return merged


def scale_confidence(conf: float, low: float = 0.5) -> float:
    if conf <= low:
        return 0.0
    return min((conf - low) / (1.0 - low), 1.0)


def get_answer_representative(answers: List[str]) -> Dict[str, str]:
    unique = []
    mapping = {}
    for ans in answers:
        found = False
        for rep in unique:
            if safe_math_equal(ans, rep):
                mapping[ans] = rep
                found = True
                break
        if not found:
            unique.append(ans)
            mapping[ans] = ans
    return mapping


# ─────────────────────────────────────────────
# 主选择器
# ─────────────────────────────────────────────

class AnswerSelector:
    """
    支持的策略：
      基础:
        first                  直接取第一个rollout
        majority               多数投票
      置信度:
        highest_confidence     取置信度最高的rollout
        weighted_confidence    按置信度加权投票
        confidence_filter      只保留高于阈值的rollout再投票
      组合 (voting + confidence + backward):
        vc_arithmetic          算术平均(voting, conf_scaled)
        vc_geometric           几何平均(voting, conf_scaled)
        vc_alpha               参数化: α*voting + (1-α)*conf_scaled
        vb_arithmetic          算术平均(voting, backward)
        vb_geometric           几何平均(voting, backward)
        vb_alpha               参数化: α*voting + (1-α)*backward
        vcb_arithmetic         算术平均(voting, conf_scaled, backward)
        vcb_geometric          几何平均(voting, conf_scaled, backward)
        vcb_alpha              参数化: α*voting + β*conf_scaled + γ*backward
        fobar                  前向+后向几何平均
        voting_then_conf       投票优先，平票时用置信度决胜
        voting_then_backward   投票优先，平票时用backward决胜
        voting_then_vcb        投票优先，平票时用conf+backward决胜

    backward_key 参数控制使用哪个 backward 字段：
        'digits'      使用 backward_result        (数字mask，按candidate聚合)
        'probability' 使用 backward_result_probability (关键片段概率，按rollout)
    """

    STRATEGIES = [
        'first', 'majority',
        'highest_confidence', 'weighted_confidence', 'confidence_filter',
        'highest_backward',  # ← 新增
        'confidence_top_half',
        'vc_arithmetic', 'vc_geometric', 'vc_alpha',
        'vb_arithmetic', 'vb_geometric', 'vb_alpha',
        'vcb_arithmetic', 'vcb_geometric', 'vcb_alpha',
        'fobar',
        'voting_then_conf', 'voting_then_backward', 'voting_then_vcb',
    ]

    def __init__(self, strategy: str = 'majority',
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 confidence_threshold: float = 0.95,
                 confidence_scale_low: float = 0.5,
                 backward_key: str = 'digits'):
        """
        backward_key: 'digits' 或 'probability'
        """
        assert strategy in self.STRATEGIES, f"Unknown strategy: {strategy}"
        assert backward_key in ('digits', 'probability'), \
            f"backward_key must be 'digits' or 'probability', got {backward_key}"
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.confidence_threshold = confidence_threshold
        self.confidence_scale_low = confidence_scale_low
        self.backward_key = backward_key
        self._parsed_cache = {}

    def _parse(self, rollout: str) -> Tuple[str, str]:
        if rollout not in self._parsed_cache:
            self._parsed_cache[rollout] = parse_ground_truth(rollout)
        return self._parsed_cache[rollout]

    def _get_answers(self, rollouts: List[str]) -> List[str]:
        return [self._parse(r)[1] for r in rollouts]

    def _normalized_voting(self, answers: List[str]) -> Dict[str, float]:
        counts = Counter(answers)
        norm = normalize_answers_in_votes(dict(counts))
        total = len(answers)
        return {ans: cnt / total for ans, cnt in norm.items()}

    def _avg_confidence_per_rollout(self, records_list: List[List[Dict]]) -> List[float]:
        result = []
        for records in records_list:
            confs = [r.get('confidence', 0.0) for r in records]
            result.append(sum(confs) / len(confs) if confs else 0.0)
        return result

    def _confidence_per_answer(self, answers: List[str],
                                conf_per_rollout: List[float],
                                mapping: Dict[str, str]) -> Dict[str, float]:
        accum: Dict[str, List[float]] = {}
        for ans, conf in zip(answers, conf_per_rollout):
            rep = mapping.get(ans, ans)
            accum.setdefault(rep, []).append(
                scale_confidence(conf, self.confidence_scale_low))
        return {rep: sum(v) / len(v) for rep, v in accum.items()}

    def _backward_per_answer_digits(self, sample: Dict,
                                     mapping: Dict[str, str]) -> Dict[str, float]:
        """
        从 backward_result 读取 backward score（数字mask版）。
        每个 backward_result 条目对应一个 candidate。
        """
        backward_result = sample.get('backward_result', [])
        scores: Dict[str, List[float]] = {}
        for br in backward_result:
            candidate = br.get('candidate', '')
            rep = mapping.get(candidate, candidate)
            scores.setdefault(rep, []).append(br.get('backward_score', 0.0))
        return {rep: sum(v) / len(v) for rep, v in scores.items()}

    def _backward_per_answer_probability(self, sample: Dict,
                                          answers: List[str],
                                          mapping: Dict[str, str]) -> Dict[str, float]:
        """
        从 backward_result_probability 读取 backward score（关键片段概率版）。
        每个条目对应一个 rollout，需要和 answers 按 rollout index 对齐。
        """
        prob_results = sample.get('backward_result_probability', [])
        if not prob_results:
            return {}

        # 按 rollout_idx 排序，保证和 answers 对齐
        sorted_results = sorted(prob_results, key=lambda x: x.get('rollout_idx', 0))

        scores: Dict[str, List[float]] = {}
        for ans, pr in zip(answers, sorted_results):
            rep = mapping.get(ans, ans)
            score = pr.get('backward_score_probability', 0.0)
            scores.setdefault(rep, []).append(score)
        return {rep: sum(v) / len(v) for rep, v in scores.items()}

    def _backward_per_answer(self, sample: Dict,
                              answers: List[str],
                              mapping: Dict[str, str]) -> Dict[str, float]:
        """统一接口，根据 backward_key 选择来源"""
        if self.backward_key == 'probability':
            return self._backward_per_answer_probability(sample, answers, mapping)
        else:
            return self._backward_per_answer_digits(sample, mapping)

    def _get_rollouts_records(self, sample: Dict) -> List[List[Dict]]:
        if 'rollouts_records' in sample:
            return sample['rollouts_records']
        confs = sample.get('rollouts_confidence', [])
        return [[{'confidence': c}] for c in confs]

    # ── 策略实现 ──────────────────────────────

    def select_answer(self, sample: Dict) -> Tuple[str, Dict]:
        rollouts = sample['rollouts']
        answers_raw = self._get_answers(rollouts)
        mapping = get_answer_representative(answers_raw)
        answers = [mapping[a] for a in answers_raw]
        voting = self._normalized_voting(answers_raw)

        s = self.strategy

        if s == 'first':
            return answers[0], {'strategy': s, 'selected_index': 0}

        if s == 'majority':
            best = max(voting, key=voting.get)
            return best, {'strategy': s, 'voting': voting}

        records_list = self._get_rollouts_records(sample)
        conf_per_rollout = self._avg_confidence_per_rollout(records_list) \
            if records_list else [0.0] * len(rollouts)

        if s == 'highest_confidence':
            best_idx = int(np.argmax(conf_per_rollout))
            return answers[best_idx], {
                'strategy': s,
                'selected_index': best_idx,
                'confidence': conf_per_rollout[best_idx]
            }

        if s == 'weighted_confidence':
            accum: Dict[str, float] = {}
            for ans, conf in zip(answers, conf_per_rollout):
                accum[ans] = accum.get(ans, 0.0) + conf
            best = max(accum, key=accum.get)
            return best, {'strategy': s, 'weighted_scores': accum, 'voting': voting}

        if s == 'confidence_filter':
            high = [a for a, c in zip(answers, conf_per_rollout)
                    if c >= self.confidence_threshold]
            if not high:
                high = answers
            cnt = Counter(high)
            best = cnt.most_common(1)[0][0]
            return best, {
                'strategy': s,
                'threshold': self.confidence_threshold,
                'filtered_count': len(high),
                'total': len(answers)
            }
        
        if s == 'confidence_top_half':
            if not conf_per_rollout or all(c == 0 for c in conf_per_rollout):
                # 没有 confidence 信息，fallback 到 majority
                best = max(voting, key=voting.get)
                return best, {'strategy': s, 'note': 'no confidence, fallback to majority'}

            # 按 confidence 排序，取前50%
            n = len(answers)
            top_k = max(1, n // 2)
            sorted_indices = sorted(range(n), key=lambda i: conf_per_rollout[i], reverse=True)
            top_indices = sorted_indices[:top_k]
            top_answers = [answers[i] for i in top_indices]

            # 在 top_answers 里做多数投票
            cnt = Counter(top_answers)
            norm = normalize_answers_in_votes(dict(cnt))
            best = max(norm, key=norm.get)

            return best, {
                'strategy': s,
                'top_k': top_k,
                'total': n,
                'top_confidences': [conf_per_rollout[i] for i in top_indices],
                'top_answers': top_answers,
                'vote_counts': dict(cnt),
            }
        
        if s == 'highest_backward':
            backward = self._backward_per_answer(sample, answers_raw, mapping)
            if not backward:
                # fallback to majority
                best = max(voting, key=voting.get)
                return best, {'strategy': s, 'note': 'no backward, fallback to majority'}
            best = max(backward, key=backward.get)
            return best, {
                'strategy': s,
                'backward_key': self.backward_key,
                'backward': backward,
            }

        conf_scaled = self._confidence_per_answer(answers, conf_per_rollout, mapping)
        backward = self._backward_per_answer(sample, answers_raw, mapping)
        unique_answers = list(voting.keys())

        def safe_b(ans):
            return max(backward.get(ans, 0.0), 1e-6)

        def safe_c(ans):
            return max(conf_scaled.get(ans, 0.0), 1e-6)

        def safe_v(ans):
            return max(voting.get(ans, 0.0), 1e-6)

        scores: Dict[str, float] = {}

        if s == 'vc_arithmetic':
            for a in unique_answers:
                scores[a] = (voting.get(a, 0) + conf_scaled.get(a, 0)) / 2

        elif s == 'vc_geometric':
            for a in unique_answers:
                scores[a] = np.sqrt(safe_v(a) * safe_c(a))

        elif s == 'vc_alpha':
            for a in unique_answers:
                scores[a] = self.alpha * voting.get(a, 0) + \
                             (1 - self.alpha) * conf_scaled.get(a, 0)

        elif s == 'vb_arithmetic':
            for a in unique_answers:
                scores[a] = (voting.get(a, 0) + backward.get(a, 0)) / 2

        elif s == 'vb_geometric':
            for a in unique_answers:
                scores[a] = np.sqrt(safe_v(a) * safe_b(a))

        elif s == 'vb_alpha':
            for a in unique_answers:
                scores[a] = self.alpha * voting.get(a, 0) + \
                             (1 - self.alpha) * backward.get(a, 0)

        elif s == 'vcb_arithmetic':
            for a in unique_answers:
                scores[a] = (voting.get(a, 0) +
                              conf_scaled.get(a, 0) +
                              backward.get(a, 0)) / 3

        elif s == 'vcb_geometric':
            for a in unique_answers:
                scores[a] = (safe_v(a) * safe_c(a) * safe_b(a)) ** (1 / 3)

        elif s == 'vcb_alpha':
            total = self.alpha + self.beta + self.gamma
            a_, b_, g_ = self.alpha / total, self.beta / total, self.gamma / total
            for a in unique_answers:
                scores[a] = (a_ * voting.get(a, 0) +
                              b_ * conf_scaled.get(a, 0) +
                              g_ * backward.get(a, 0))

        elif s == 'fobar':
            for a in unique_answers:
                scores[a] = np.sqrt(safe_v(a) * safe_b(a))

        elif s == 'voting_then_conf':
            max_votes = max(voting.values())
            tied = [a for a, v in voting.items() if v == max_votes]
            if len(tied) == 1:
                scores = {tied[0]: 1.0}
            else:
                for a in tied:
                    scores[a] = conf_scaled.get(a, 0.0)

        elif s == 'voting_then_backward':
            max_votes = max(voting.values())
            tied = [a for a, v in voting.items() if v == max_votes]
            if len(tied) == 1:
                scores = {tied[0]: 1.0}
            else:
                for a in tied:
                    scores[a] = backward.get(a, 0.0)

        elif s == 'voting_then_vcb':
            max_votes = max(voting.values())
            tied = [a for a, v in voting.items() if v == max_votes]
            if len(tied) == 1:
                scores = {tied[0]: 1.0}
            else:
                total = self.alpha + self.beta + self.gamma
                a_, b_, g_ = self.alpha / total, self.beta / total, self.gamma / total
                for a in tied:
                    scores[a] = (a_ * voting.get(a, 0) +
                                 b_ * conf_scaled.get(a, 0) +
                                 g_ * backward.get(a, 0))

        best = max(scores, key=scores.get) if scores else (unique_answers[0] if unique_answers else '')

        return best, {
            'strategy': s,
            'backward_key': self.backward_key,
            'voting': voting,
            'conf_scaled': conf_scaled,
            'backward': backward,
            'combined_scores': scores,
        }


# ─────────────────────────────────────────────
# 评估工具
# ─────────────────────────────────────────────

def evaluate_single(args_tuple):
    sample, strategy, gt_key, kwargs, idx = args_tuple
    selector = AnswerSelector(strategy=strategy, **kwargs)
    try:
        selected, info = selector.select_answer(sample)
    except Exception as e:
        traceback.print_exc()
        selected, info = '', {'error': str(e)}
    gt = sample.get(gt_key, '')
    return {
        'sample_idx': idx,
        'strategy': strategy,
        'selected_answer': selected,
        'ground_truth': gt,
        'is_correct': safe_math_equal(selected, gt),
        'details': info,
    }


def evaluate_dataset(dataset: List[Dict],
                     strategies: List[str],
                     gt_key: str = 'answer',
                     num_workers: int = None,
                     sequential: bool = False,
                     **kwargs) -> Dict[str, Any]:
    results = {}
    workers = num_workers or cpu_count()

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy}  (backward_key={kwargs.get('backward_key','digits')})")
        print(f"{'=' * 60}")
        args_list = [(s, strategy, gt_key, kwargs, i)
                     for i, s in enumerate(dataset)]
        t0 = time.time()

        if sequential:
            raw = [evaluate_single(a) for a in tqdm(args_list, desc=strategy)]
        else:
            raw = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(evaluate_single, a): i
                           for i, a in enumerate(args_list)}
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(args_list), desc=strategy, ncols=80):
                    try:
                        result = future.result(timeout=10)
                        raw.append(result)
                    except FuturesTimeout:
                        idx = futures[future]
                        print(f"\n⚠️  Sample {idx} timed out")
                        raw.append({
                            'sample_idx': idx, 'strategy': strategy,
                            'selected_answer': '', 'ground_truth': args_list[idx][2],
                            'is_correct': False, 'details': {'error': 'timeout'},
                        })
                    except Exception as e:
                        idx = futures[future]
                        raw.append({
                            'sample_idx': idx, 'strategy': strategy,
                            'selected_answer': '', 'ground_truth': args_list[idx][2],
                            'is_correct': False, 'details': {'error': str(e)},
                        })

        elapsed = time.time() - t0
        raw.sort(key=lambda x: x['sample_idx'])
        for r in raw:
            del r['sample_idx']

        correct = sum(1 for r in raw if r['is_correct'])
        acc = correct / len(dataset) if dataset else 0
        results[strategy] = {
            'accuracy': acc, 'correct': correct,
            'total': len(dataset), 'elapsed': elapsed, 'details': raw,
        }
        print(f"{strategy}: {correct}/{len(dataset)} = {acc * 100:.2f}%  ({elapsed:.1f}s)")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--strategies', '-s', nargs='+',
                        default=['first', 'majority', 'weighted_confidence',
                                 'vcb_geometric', 'vcb_alpha', 'voting_then_vcb'],
                        choices=AnswerSelector.STRATEGIES)
    parser.add_argument('--gt_key', default='answer')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--confidence_threshold', type=float, default=0.95)
    parser.add_argument('--confidence_scale_low', type=float, default=0.5)
    parser.add_argument('--backward_key', type=str, default='digits',
                        choices=['digits', 'probability'],
                        help=(
                            'digits      : 使用 backward_result (数字mask)\n'
                            'probability : 使用 backward_result_probability (关键片段概率)'
                        ))
    parser.add_argument('--num_workers', '-n', type=int, default=None)
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()

    print(f"Loading {args.input_file} ...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples")

    # 检查 backward 数据是否存在
    needs_backward = {
        'vb_arithmetic', 'vb_geometric', 'vb_alpha',
        'vcb_arithmetic', 'vcb_geometric', 'vcb_alpha',
        'fobar', 'voting_then_backward', 'voting_then_vcb'
    }

    if args.backward_key == 'digits':
        has_backward = any('backward_result' in s and s['backward_result']
                           for s in dataset)
        bwd_field = 'backward_result'
    else:
        has_backward = any('backward_result_probability' in s and s['backward_result_probability']
                           for s in dataset)
        bwd_field = 'backward_result_probability'

    if not has_backward:
        removed = [s for s in args.strategies if s in needs_backward]
        if removed:
            print(f"⚠️  No {bwd_field} found, removing: {removed}")
            args.strategies = [s for s in args.strategies if s not in needs_backward]

    selector_kwargs = dict(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        confidence_threshold=args.confidence_threshold,
        confidence_scale_low=args.confidence_scale_low,
        backward_key=args.backward_key,
    )

    results = evaluate_dataset(
        dataset,
        strategies=args.strategies,
        gt_key=args.gt_key,
        num_workers=args.num_workers,
        sequential=args.sequential,
        **selector_kwargs,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = {}
    for strat, data in results.items():
        summary[strat] = {
            'accuracy': data['accuracy'],
            'correct': data['correct'],
            'total': data['total'],
        }
        print(f"{strat:30s} {data['correct']:4d}/{data['total']} = {data['accuracy'] * 100:.2f}%")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'details': results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output_file}")


if __name__ == '__main__':
    main()