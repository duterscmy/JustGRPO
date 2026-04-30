import json
import argparse
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import sys, os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.grader import math_equal
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
                # correct = int(math_equal(ans, gt))
                correct = int(ans.strip() == gt.strip())
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
                # correct = int(math_equal(ans, gt))
                correct = int(ans.strip() == gt.strip())
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
                # correct = int(math_equal(ans, gt))
                correct = int(ans.strip() == gt.strip())
            except Exception:
                correct = 0
            pairs.append((float(score), correct))
    return pairs


# ─────────────────────────────────────────────
# 分桶 + 统计 (默认 6 个桶)
# ─────────────────────────────────────────────

def bucket_accuracy(pairs: List[Tuple[float, int]],
                    n_buckets: int = 6) -> Tuple[List[str], List[float], List[int], List[float]]:
    """
    按分位数分桶，确保每桶样本数量大致持平。
    返回 (bucket_labels, accuracies, counts, bucket_centers)
    """
    if not pairs:
        return [], [], [], []

    # 按 score 排序
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    scores = np.array([p[0] for p in sorted_pairs])
    labels = np.array([p[1] for p in sorted_pairs])

    # 用 np.array_split 均匀切分，每桶数量尽量持平
    indices = np.array_split(np.arange(len(sorted_pairs)), n_buckets)

    bucket_labels = []
    accuracies = []
    counts = []
    bucket_centers = []  # 用于趋势线

    for idx in indices:
        if len(idx) == 0:
            continue
        bucket_scores = scores[idx]
        bucket_labels_arr = labels[idx]
        lo, hi = bucket_scores.min(), bucket_scores.max()
        acc = bucket_labels_arr.mean() * 100
        
        # 左闭右开区间格式
        bucket_labels.append(f"[{lo:.3f}, {hi:.3f})")
        accuracies.append(acc)
        counts.append(len(idx))
        # 使用区间中点作为趋势线的 x 坐标
        bucket_centers.append((lo + hi) / 2)

    return bucket_labels, accuracies, counts, bucket_centers


# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────

def plot_single_ax(ax, bucket_labels: List[str],
                   accuracies: List[float],
                   counts: List[int],
                   bucket_centers: List[float],
                   title: str):
    """
    在指定的 ax 上绘制柱状图 + 趋势折线
    """
    if not bucket_labels:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    x = np.arange(len(bucket_labels))
    
    # 柱状图
    bars = ax.bar(x, accuracies, color="#1579C0", alpha=0.8, width=0.6, 
                  edgecolor='white', linewidth=0.5, zorder=2)

    # 趋势折线
    ax.plot(x, accuracies, color='#E67E22', marker='o', markersize=5, 
            linewidth=2, linestyle='-', zorder=3, label='Trend')

    # 坐标轴标签
    ax.set_xlabel('Rollout confidence', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xticks(x)
    # 横坐标不倾斜，水平显示
    ax.set_xticklabels(bucket_labels, fontsize=10, rotation=0, ha='center')
    
    # 纵坐标范围
    y_min = max(0, min(accuracies) - 8)
    y_max = min(105, max(accuracies) + 8)
    if y_max - y_min < 30:
        y_min = max(0, y_min - 10)
        y_max = min(105, y_max + 10)
    ax.set_ylim(y_min, y_max)

    # 在柱子上方显示准确率
    y_offset = (y_max - y_min) * 0.02
    for bar, acc, cnt in zip(bars, accuracies, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=12, 
                    color='#E67E22', fontweight='medium')

    # 轻微网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC', zorder=0)
    ax.set_axisbelow(True)

    # 去掉顶部和右侧 spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # 标题和图例
    # ax.set_title(title, fontsize=11, fontweight='medium')
    ax.legend(loc='lower right', fontsize=8)


def plot_multiple_metrics(input_files: List[str],
                          metric_name: str,
                          n_buckets: int,
                          save_path: Optional[str],
                          dataset_names: Optional[List[str]] = None):
    """
    为多个 JSON 文件绘制同一指标的对比图
    """
    n_files = len(input_files)
    
    # 自动确定子图布局
    if n_files == 1:
        n_rows, n_cols = 1, 1
        figsize = (8, 4)
    elif n_files == 2:
        n_rows, n_cols = 1, 2
        figsize = (14, 5)
    elif n_files <= 4:
        n_rows, n_cols = 2, 2
        figsize = (14, 10)
    else:
        n_rows = (n_files + 2) // 3
        n_cols = 3
        figsize = (15, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 确保 axes 是一维数组
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 隐藏多余的子图
    for i in range(n_files, len(axes)):
        axes[i].set_visible(False)
    
    # 为每个文件绘图
    for idx, (input_file, ax) in enumerate(zip(input_files, axes)):
        print(f"\nProcessing {input_file} ...")
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"  Loaded {len(dataset)} samples")
        
        # 提取数据
        extractor_map = {
            'confidence': extract_confidence_pairs,
            'backward_digits': extract_backward_digits_pairs,
            'backward_probability': extract_backward_probability_pairs,
        }
        
        pairs = extractor_map[metric_name](dataset)
        print(f"  → {len(pairs)} rollout pairs extracted")
        
        if not pairs:
            ax.text(0.5, 0.5, f'No data\n{os.path.basename(input_file)}', 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        # 计算统计信息
        scores = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        corr = np.corrcoef(scores, labels)[0, 1]
        overall_acc = np.mean(labels) * 100
        
        print(f"  overall_acc={overall_acc:.2f}%, Pearson_r={corr:.4f}")
        
        # 分桶
        bucket_labels, accuracies, counts, bucket_centers = bucket_accuracy(pairs, n_buckets)
        
        # 标题：使用文件名或自定义名称
        if dataset_names and idx < len(dataset_names):
            title = dataset_names[idx]
        else:
            title = os.path.basename(input_file).replace('.json', '').replace('_', ' ')
        
        # 绘图
        plot_single_ax(ax, bucket_labels, accuracies, counts, bucket_centers, title)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/{metric_name}.pdf", dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\nSaved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot score-vs-accuracy correlation for confidence/backward metrics')
    parser.add_argument('input_files', nargs='+', help='Input JSON file(s)')
    parser.add_argument('--metric', '-m', type=str, default='confidence',
                        choices=['confidence', 'backward_digits', 'backward_probability'],
                        help='Which metric to plot (default: confidence)')
    parser.add_argument('--n_buckets', type=int, default=6,
                        help='Number of buckets (default: 6)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the combined plot')
    parser.add_argument('--dataset_names', '-n', nargs='+', type=str, default=None,
                        help='Custom names for each dataset (order matches input files)')
    args = parser.parse_args()

    plot_multiple_metrics(
        input_files=args.input_files,
        metric_name=args.metric,
        n_buckets=args.n_buckets,
        save_path=args.save_path,
        dataset_names=args.dataset_names
    )


if __name__ == '__main__':
    main()