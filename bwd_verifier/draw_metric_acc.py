import json
import argparse
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys, os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grader import math_equal
from utils.parser import parse_ground_truth


# ─────────────────────────────────────────────
# 提取各指标的 (score, is_correct) 对
# ─────────────────────────────────────────────

def extract_confidence_pairs(dataset: List[Dict]) -> List[Tuple[float, int]]:
    """
    从 rollouts_confidence 或 rollouts_records 提取每个 rollout 的
    (avg_confidence, is_correct) 对
    """
    pairs = []
    for sample in dataset:
        rollouts = sample.get('rollouts', [])
        gt = sample.get('answer', '')

        # 支持两种格式
        confs = sample.get('rollouts_confidence', [])
        if not confs and 'rollouts_records' in sample:
            confs = []
            for records in sample['rollouts_records']:
                cs = [r.get('confidence', 0.0) for r in records]
                confs.append(sum(cs) / len(cs) if cs else 0.0)

        for rollout, conf in zip(rollouts, confs):
            _, ans = parse_ground_truth(rollout)
            try:
                correct = int(math_equal(ans, gt))
            except Exception:
                correct = 0
            pairs.append((float(conf), correct))
    return pairs


def extract_backward_digits_pairs(dataset: List[Dict]) -> List[Tuple[float, int]]:
    """
    从 backward_result 提取每个 rollout 的
    (backward_score, is_correct) 对
    """
    pairs = []
    for sample in dataset:
        rollouts = sample.get('rollouts', [])
        gt = sample.get('answer', '')
        backward_results = sample.get('backward_result', [])

        for rollout, br in zip(rollouts, backward_results):
            score = br.get('backward_score', None)
            if score is None:
                continue
            _, ans = parse_ground_truth(rollout)
            try:
                correct = int(math_equal(ans, gt))
            except Exception:
                correct = 0
            pairs.append((float(score), correct))
    return pairs


def extract_backward_probability_pairs(dataset: List[Dict]) -> List[Tuple[float, int]]:
    """
    从 backward_result_probability 提取每个 rollout 的
    (backward_score_probability, is_correct) 对
    """
    pairs = []
    for sample in dataset:
        rollouts = sample.get('rollouts', [])
        gt = sample.get('answer', '')
        prob_results = sample.get('backward_result_probability', [])
        if not prob_results:
            continue

        sorted_results = sorted(prob_results, key=lambda x: x.get('rollout_idx', 0))

        for rollout, pr in zip(rollouts, sorted_results):
            score = pr.get('backward_score_probability', None)
            if score is None:
                continue
            _, ans = parse_ground_truth(rollout)
            try:
                correct = int(math_equal(ans, gt))
            except Exception:
                correct = 0
            pairs.append((float(score), correct))
    return pairs


# ─────────────────────────────────────────────
# 分桶 + 统计
# ─────────────────────────────────────────────

def bucket_accuracy(pairs: List[Tuple[float, int]],
                    n_buckets: int = 10) -> Tuple[List[str], List[float], List[int]]:
    """
    按 score 均匀分桶，统计每桶的准确率和样本数。
    返回 (bucket_labels, accuracies, counts)
    """
    if not pairs:
        return [], [], []

    scores = np.array([p[0] for p in pairs])
    labels = np.array([p[1] for p in pairs])

    min_s, max_s = scores.min(), scores.max()
    # 边界稍微扩一点，防止最大值落在桶外
    edges = np.linspace(min_s, max_s + 1e-9, n_buckets + 1)

    bucket_labels = []
    accuracies = []
    counts = []

    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        cnt = mask.sum()
        acc = labels[mask].mean() * 100 if cnt > 0 else 0.0
        bucket_labels.append(f"{lo:.3f}\n–{hi:.3f}")
        accuracies.append(acc)
        counts.append(int(cnt))

    return bucket_labels, accuracies, counts


# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────

def plot_buckets(bucket_labels: List[str],
                 accuracies: List[float],
                 counts: List[int],
                 metric_name: str,
                 save_path: str = None,
                 dataset_name: str = ''):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = np.arange(len(bucket_labels))
    bars = ax1.bar(x, accuracies, color='steelblue', alpha=0.8, width=0.6)

    ax1.set_xlabel(f'{metric_name} bucket', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bucket_labels, fontsize=8)
    ax1.set_ylim(0, 110)

    # 在柱子上方显示准确率
    for bar, acc, cnt in zip(bars, accuracies, counts):
        if cnt > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5,
                     f'{acc:.1f}%',
                     ha='center', va='bottom', fontsize=8, color='steelblue')

    # 右轴显示样本数
    ax2 = ax1.twinx()
    ax2.plot(x, counts, 'o--', color='coral', linewidth=1.5, markersize=5, label='count')
    ax2.set_ylabel('Sample count', fontsize=12, color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    title = f'{metric_name} vs Accuracy'
    if dataset_name:
        title += f'  [{dataset_name}]'
    ax1.set_title(title, fontsize=13)

    # 计算整体 Pearson 相关系数
    # valid = [(s, l) for s, l in zip(
    #     [float(b.replace('\n–', '').split('–')[0]) for b in bucket_labels],
    #     accuracies
    # ) if counts[bucket_labels.index(b)] > 0
    #     for b in [bucket_labels[bucket_labels.index(b)]]]

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_all_metrics(pairs_dict: Dict[str, List[Tuple[float, int]]],
                     n_buckets: int,
                     save_dir: str,
                     dataset_name: str):
    """为每个指标单独画图，并输出 Pearson 相关系数"""
    for metric_name, pairs in pairs_dict.items():
        if not pairs:
            print(f"[{metric_name}] No data, skipping.")
            continue

        scores = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

        # Pearson 相关系数
        corr = np.corrcoef(scores, labels)[0, 1]
        print(f"[{metric_name}] n={len(pairs)}, "
              f"mean_score={np.mean(scores):.4f}, "
              f"overall_acc={np.mean(labels)*100:.2f}%, "
              f"Pearson_r={corr:.4f}")

        bucket_labels, accuracies, counts = bucket_accuracy(pairs, n_buckets)

        fname = metric_name.replace(' ', '_').replace('/', '_') + '.png'
        save_path = os.path.join(save_dir, fname) if save_dir else None
        plot_buckets(bucket_labels, accuracies, counts,
                     metric_name=metric_name,
                     save_path=save_path,
                     dataset_name=dataset_name)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot score-vs-accuracy correlation for confidence/backward metrics')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('--metrics', '-m', nargs='+',
                        default=['confidence', 'backward_digits', 'backward_probability'],
                        choices=['confidence', 'backward_digits', 'backward_probability'],
                        help='Which metrics to plot')
    parser.add_argument('--n_buckets', type=int, default=10,
                        help='Number of buckets (default: 10)')
    parser.add_argument('--save_dir', type=str, default="./figs",
                        help='Directory to save plots (default: show interactively)')
    parser.add_argument('--dataset_name', type=str, default='',
                        help='Dataset name shown in plot title')
    args = parser.parse_args()

    print(f"Loading {args.input_file} ...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    extractor_map = {
        'confidence':            extract_confidence_pairs,
        'backward_digits':       extract_backward_digits_pairs,
        'backward_probability':  extract_backward_probability_pairs,
    }

    pairs_dict = {}
    for metric in args.metrics:
        print(f"Extracting {metric} ...")
        pairs = extractor_map[metric](dataset)
        pairs_dict[metric] = pairs
        print(f"  → {len(pairs)} rollout pairs extracted")

    plot_all_metrics(pairs_dict,
                     n_buckets=args.n_buckets,
                     save_dir=args.save_dir,
                     dataset_name=args.dataset_name)


if __name__ == '__main__':
    main()