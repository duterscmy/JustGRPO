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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grader import math_equal
from utils.parser import extract_answer, parse_ground_truth


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
            if math_equal(answers[i], answers[j]):
                equiv.append(answers[j])
                used.add(j)
        rep = max(equiv, key=lambda x: votes.get(x, 0))
        merged[rep] = sum(votes.get(a, 0) for a in equiv)
        used.add(i)
    return merged


def scale_confidence(conf: float, low: float = 0.5) -> float:
    """将 confidence 从 [low, 1.0] 线性拉伸到 [0, 1]"""
    if conf <= low:
        return 0.0
    return min((conf - low) / (1.0 - low), 1.0)


def get_answer_representative(answers: List[str]) -> Dict[str, str]:
    """为每个答案找等价代表元，返回 {answer: representative} 映射"""
    unique = []
    mapping = {}
    for ans in answers:
        found = False
        for rep in unique:
            if math_equal(ans, rep):
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
        fobar                  前向+后向几何平均（原始实现）
        voting_then_conf       投票优先，平票时用置信度决胜
        voting_then_backward   投票优先，平票时用backward决胜
        voting_then_vcb        投票优先，平票时用conf+backward决胜
    """

    STRATEGIES = [
        'first', 'majority',
        'highest_confidence', 'weighted_confidence', 'confidence_filter',
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
                 confidence_scale_low: float = 0.5):
        """
        Args:
            alpha: voting 权重 (vc_alpha / vb_alpha / vcb_alpha)
            beta:  confidence 权重 (vcb_alpha)
            gamma: backward 权重 (vcb_alpha)，若 alpha+beta+gamma != 1 会自动归一化
            confidence_threshold: confidence_filter 策略的阈值
            confidence_scale_low: scale_confidence 的下界（通常设 0.5 或 0.9）
        """
        assert strategy in self.STRATEGIES, f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.confidence_threshold = confidence_threshold
        self.confidence_scale_low = confidence_scale_low
        self._parsed_cache = {}

    def _parse(self, rollout: str) -> Tuple[str, str]:
        if rollout not in self._parsed_cache:
            self._parsed_cache[rollout] = parse_ground_truth(rollout)
        return self._parsed_cache[rollout]

    # ── 内部工具 ──────────────────────────────

    def _get_answers(self, rollouts: List[str]) -> List[str]:
        return [self._parse(r)[1] for r in rollouts]

    def _normalized_voting(self, answers: List[str]) -> Dict[str, float]:
        """返回每个代表元的投票占比 {rep: ratio}"""
        counts = Counter(answers)
        norm = normalize_answers_in_votes(dict(counts))
        total = len(answers)
        return {ans: cnt / total for ans, cnt in norm.items()}

    def _avg_confidence_per_rollout(self, records_list: List[List[Dict]]) -> List[float]:
        """每个 rollout 的平均 confidence"""
        result = []
        for records in records_list:
            confs = [r.get('confidence', 0.0) for r in records]
            result.append(sum(confs) / len(confs) if confs else 0.0)
        return result

    def _confidence_per_answer(self, rollouts: List[str],
                                answers: List[str],
                                conf_per_rollout: List[float],
                                mapping: Dict[str, str]) -> Dict[str, float]:
        """按代表元累加 scaled confidence，返回平均值"""
        accum: Dict[str, List[float]] = {}
        for ans, conf in zip(answers, conf_per_rollout):
            rep = mapping.get(ans, ans)
            accum.setdefault(rep, []).append(scale_confidence(conf, self.confidence_scale_low))
        return {rep: sum(v) / len(v) for rep, v in accum.items()}

    def _backward_per_answer(self, sample: Dict,
                              mapping: Dict[str, str]) -> Dict[str, float]:
        """从 backward_result 读取 backward score，按代表元映射"""
        backward_result = sample.get('backward_result', [])
        scores: Dict[str, List[float]] = {}
        for br in backward_result:
            candidate = br.get('candidate', '')
            rep = mapping.get(candidate, candidate)
            scores.setdefault(rep, []).append(br.get('backward_score', 0.0))
        return {rep: sum(v) / len(v) for rep, v in scores.items()}

    def _get_rollouts_records(self, sample: Dict) -> List[List[Dict]]:
        """兼容 rollouts_records 和 rollouts_confidence 两种格式"""
        if 'rollouts_records' in sample:
            return sample['rollouts_records']
        # 如果只有 rollouts_confidence（已转换格式），构造伪 records
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

        # ── 基础策略 ──
        if s == 'first':
            return answers[0], {'strategy': s, 'selected_index': 0}

        if s == 'majority':
            best = max(voting, key=voting.get)
            return best, {'strategy': s, 'voting': voting}

        # ── 置信度策略 ──
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
                high = answers  # fallback
            cnt = Counter(high)
            best = cnt.most_common(1)[0][0]
            return best, {
                'strategy': s,
                'threshold': self.confidence_threshold,
                'filtered_count': len(high),
                'total': len(answers)
            }

        # ── 组合策略（需要 conf_scaled 和/或 backward） ──
        conf_scaled = self._confidence_per_answer(rollouts, answers_raw,
                                                  conf_per_rollout, mapping)
        backward = self._backward_per_answer(sample, mapping)
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
                scores[a] = (safe_v(a) * safe_c(a) * safe_b(a)) ** (1/3)

        elif s == 'vcb_alpha':
            total = self.alpha + self.beta + self.gamma
            a_, b_, g_ = self.alpha/total, self.beta/total, self.gamma/total
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
                a_, b_, g_ = self.alpha/total, self.beta/total, self.gamma/total
                for a in tied:
                    scores[a] = (a_ * voting.get(a, 0) +
                                 b_ * conf_scaled.get(a, 0) +
                                 g_ * backward.get(a, 0))

        if not scores:
            best = unique_answers[0] if unique_answers else ''
        else:
            best = max(scores, key=scores.get)

        return best, {
            'strategy': s,
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
        'is_correct': math_equal(selected, gt),
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
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")
        args_list = [(s, strategy, gt_key, kwargs, i)
                     for i, s in enumerate(dataset)]
        t0 = time.time()

        if sequential:
            raw = [evaluate_single(a) for a in tqdm(args_list, desc=strategy)]
        else:
            with Pool(workers) as pool:
                raw = list(tqdm(
                    pool.imap_unordered(evaluate_single, args_list),
                    total=len(dataset), desc=strategy, ncols=80))

        elapsed = time.time() - t0
        raw.sort(key=lambda x: x['sample_idx'])
        for r in raw:
            del r['sample_idx']

        correct = sum(1 for r in raw if r['is_correct'])
        acc = correct / len(dataset) if dataset else 0
        results[strategy] = {
            'accuracy': acc,
            'correct': correct,
            'total': len(dataset),
            'elapsed': elapsed,
            'details': raw,
        }
        print(f"{strategy}: {correct}/{len(dataset)} = {acc*100:.2f}%  ({elapsed:.1f}s)")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--strategies', '-s', nargs='+',
                        default=['majority', 'weighted_confidence',
                                 'vcb_geometric', 'vcb_alpha',
                                 'voting_then_vcb'],
                        choices=AnswerSelector.STRATEGIES)
    parser.add_argument('--gt_key', default='answer')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='voting weight for *_alpha strategies')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='confidence weight for vcb_alpha / voting_then_vcb')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='backward weight for vcb_alpha / voting_then_vcb')
    parser.add_argument('--confidence_threshold', type=float, default=0.95)
    parser.add_argument('--confidence_scale_low', type=float, default=0.5,
                        help='lower bound for confidence scaling (default 0.5)')
    parser.add_argument('--num_workers', '-n', type=int, default=None)
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()

    print(f"Loading {args.input_file} ...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples")

    # 自动过滤需要 backward 但没有数据的策略
    needs_backward = {'vb_arithmetic', 'vb_geometric', 'vb_alpha',
                      'vcb_arithmetic', 'vcb_geometric', 'vcb_alpha',
                      'fobar', 'voting_then_backward', 'voting_then_vcb'}
    has_backward = any('backward_result' in s and s['backward_result']
                       for s in dataset)
    if not has_backward:
        removed = [s for s in args.strategies if s in needs_backward]
        if removed:
            print(f"⚠️  No backward_result found, removing: {removed}")
            args.strategies = [s for s in args.strategies
                               if s not in needs_backward]

    selector_kwargs = dict(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        confidence_threshold=args.confidence_threshold,
        confidence_scale_low=args.confidence_scale_low,
    )

    results = evaluate_dataset(
        dataset,
        strategies=args.strategies,
        gt_key=args.gt_key,
        num_workers=args.num_workers,
        sequential=args.sequential,
        **selector_kwargs,
    )

    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = {}
    for strat, data in results.items():
        summary[strat] = {
            'accuracy': data['accuracy'],
            'correct': data['correct'],
            'total': data['total'],
        }
        print(f"{strat:30s} {data['correct']:4d}/{data['total']} = {data['accuracy']*100:.2f}%")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'details': results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output_file}")


if __name__ == '__main__':
    main()