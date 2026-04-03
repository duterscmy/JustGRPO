import json
import numpy as np
from collections import defaultdict

def calculate_accuracies(results_file_path):
    """
    计算各种准确率指标
    
    Args:
        results_file_path: FOBAR结果JSON文件路径
    """
    with open(results_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 统计变量
    total_samples = len(results)
    
    # 1. 原始数据中的 best_is_correct (来自statistics)
    best_is_correct_values = []
    for r in results:
        if 'original_sample' in r and 'statistics' in r['original_sample']:
            best_is_correct = r['original_sample']['statistics'].get('best_is_correct')
            if best_is_correct is not None:
                best_is_correct_values.append(best_is_correct)
    
    # 2. FOBAR结果的准确率
    geometric_correct = []
    arithmetic_correct = []
    
    for r in results:
        if 'fobar_results' in r:
            geometric_correct.append(r['fobar_results']['geometric']['is_correct'])
            arithmetic_correct.append(r['fobar_results']['arithmetic']['is_correct'])
    
    # 3. 每个rollout的准确率 (如果有is_correct字段)
    rollout_correct = []
    for r in results:
        if 'original_sample' in r and 'rollouts' in r['original_sample']:
            for rollout in r['original_sample']['rollouts']:
                if 'is_correct' in rollout:
                    rollout_correct.append(rollout['is_correct'])
    
    # 计算均值
    print("="*80)
    print("准确率统计")
    print("="*80)
    
    if best_is_correct_values:
        best_is_correct_mean = np.mean(best_is_correct_values)
        print(f"\n1. 原始数据 best_is_correct (多数投票正确率):")
        print(f"   正确数: {sum(best_is_correct_values)}/{len(best_is_correct_values)}")
        print(f"   准确率: {best_is_correct_mean:.4f} ({best_is_correct_mean*100:.2f}%)")
    
    if geometric_correct:
        geometric_mean = np.mean(geometric_correct)
        print(f"\n2. FOBAR Geometric Mean 准确率:")
        print(f"   正确数: {sum(geometric_correct)}/{len(geometric_correct)}")
        print(f"   准确率: {geometric_mean:.4f} ({geometric_mean*100:.2f}%)")
    
    if arithmetic_correct:
        arithmetic_mean = np.mean(arithmetic_correct)
        print(f"\n3. FOBAR Arithmetic Mean 准确率:")
        print(f"   正确数: {sum(arithmetic_correct)}/{len(arithmetic_correct)}")
        print(f"   准确率: {arithmetic_mean:.4f} ({arithmetic_mean*100:.2f}%)")
    
    if rollout_correct:
        rollout_mean = np.mean(rollout_correct)
        print(f"\n4. 单个Rollout准确率:")
        print(f"   正确数: {sum(rollout_correct)}/{len(rollout_correct)}")
        print(f"   准确率: {rollout_mean:.4f} ({rollout_mean*100:.2f}%)")
    
    # 额外统计：对比提升
    if best_is_correct_values and geometric_correct:
        improvement = geometric_mean - best_is_correct_mean
        print(f"\n5. FOBAR Geometric vs 原始多数投票:")
        print(f"   提升: {improvement:.4f} ({improvement*100:.2f}%)")
    
    if best_is_correct_values and arithmetic_correct:
        improvement = arithmetic_mean - best_is_correct_mean
        print(f"   FOBAR Arithmetic vs 原始多数投票:")
        print(f"   提升: {improvement:.4f} ({improvement*100:.2f}%)")
    
    # 详细统计
    print("\n" + "="*80)
    print("详细统计")
    print("="*80)
    
    # 按样本分析
    print("\n每个样本的对比:")
    print("-"*60)
    for i, r in enumerate(results[:10]):  # 只显示前10个
        sample_info = f"Sample {i+1}: "
        
        if 'original_sample' in r and 'statistics' in r['original_sample']:
            best_correct = r['original_sample']['statistics'].get('best_is_correct', 'N/A')
            sample_info += f"Best={best_correct} | "
        
        if 'fobar_results' in r:
            geo_correct = r['fobar_results']['geometric']['is_correct']
            arith_correct = r['fobar_results']['arithmetic']['is_correct']
            sample_info += f"Geo={geo_correct} | Arith={arith_correct}"
        
        print(sample_info)
    
    # 返回统计结果
    stats = {
        'total_samples': total_samples,
        'best_is_correct': {
            'mean': np.mean(best_is_correct_values) if best_is_correct_values else None,
            'correct': sum(best_is_correct_values) if best_is_correct_values else None,
            'total': len(best_is_correct_values) if best_is_correct_values else None
        },
        'geometric_mean': {
            'mean': np.mean(geometric_correct) if geometric_correct else None,
            'correct': sum(geometric_correct) if geometric_correct else None,
            'total': len(geometric_correct) if geometric_correct else None
        },
        'arithmetic_mean': {
            'mean': np.mean(arithmetic_correct) if arithmetic_correct else None,
            'correct': sum(arithmetic_correct) if arithmetic_correct else None,
            'total': len(arithmetic_correct) if arithmetic_correct else None
        },
        'rollout_accuracy': {
            'mean': np.mean(rollout_correct) if rollout_correct else None,
            'correct': sum(rollout_correct) if rollout_correct else None,
            'total': len(rollout_correct) if rollout_correct else None
        }
    }
    
    return stats


def calculate_accuracy_by_confidence(results_file_path):
    """
    按置信度分组计算准确率
    """
    with open(results_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 收集每个样本的置信度和是否正确
    geo_data = []
    arith_data = []
    
    for r in results:
        if 'fobar_results' in r:
            geo = r['fobar_results']['geometric']
            arith = r['fobar_results']['arithmetic']
            
            geo_data.append({
                'score': geo['selected_score'],
                'correct': geo['is_correct']
            })
            arith_data.append({
                'score': arith['selected_score'],
                'correct': arith['is_correct']
            })
    
    # 按置信度分组
    def group_by_confidence(data, bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0]):
        groups = defaultdict(list)
        for item in data:
            score = item['score']
            for i in range(len(bins)-1):
                if bins[i] <= score < bins[i+1]:
                    groups[f"{bins[i]:.1f}-{bins[i+1]:.1f}"].append(item['correct'])
                    break
            if score >= bins[-1]:
                groups[f"{bins[-1]:.1f}+"].append(item['correct'])
        return groups
    
    print("\n" + "="*80)
    print("按置信度分组的准确率")
    print("="*80)
    
    print("\nGeometric Mean:")
    geo_groups = group_by_confidence(geo_data)
    for group, corrects in sorted(geo_groups.items()):
        acc = np.mean(corrects)
        print(f"  {group}: {sum(corrects)}/{len(corrects)} = {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nArithmetic Mean:")
    arith_groups = group_by_confidence(arith_data)
    for group, corrects in sorted(arith_groups.items()):
        acc = np.mean(corrects)
        print(f"  {group}: {sum(corrects)}/{len(corrects)} = {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算FOBAR结果的准确率统计')
    parser.add_argument('results_file', type=str, help='FOBAR结果JSON文件路径')
    parser.add_argument('--confidence', action='store_true', help='按置信度分组分析')
    
    args = parser.parse_args()
    
    # 计算基本统计
    stats = calculate_accuracies(args.results_file)
    
    # 如果需要按置信度分析
    if args.confidence:
        calculate_accuracy_by_confidence(args.results_file)