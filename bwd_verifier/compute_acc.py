import json
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

def detailed_analysis(results_file_path):
    """
    详细分析FOBAR结果
    
    Args:
        results_file_path: FOBAR结果JSON文件路径
    """
    with open(results_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("="*100)
    print("FOBAR 详细分析报告")
    print("="*100)
    
    # ============ (1) 找出FOBAR反而错误的样本 ============
    print("\n" + "="*100)
    print("(1) FOBAR反而错误的样本分析")
    print("="*100)
    
    worsened_samples = []
    
    for idx, r in enumerate(results):
        if 'original_sample' in r and 'statistics' in r['original_sample']:
            best_is_correct = r['original_sample']['statistics'].get('best_is_correct', 0)
            
            if 'fobar_results' in r:
                geo_correct = r['fobar_results']['geometric']['is_correct']
                arith_correct = r['fobar_results']['arithmetic']['is_correct']
                
                # 原始正确但FOBAR错误的样本
                if best_is_correct == 1 and (geo_correct == 0 or arith_correct == 0):
                    worsened_samples.append({
                        'index': idx,
                        'ground_truth': r['original_sample'].get('ground_truth_answer'),
                        'best_is_correct': best_is_correct,
                        'geo_correct': geo_correct,
                        'arith_correct': arith_correct,
                        'forward_scores': r.get('forward_scores', {}),
                        'backward_scores': r.get('backward_scores', {}),
                        'backward_details': r.get('backward_details', {})
                    })
    
    print(f"\n发现 {len(worsened_samples)} 个样本原始多数投票正确，但FOBAR反而错误")
    
    if worsened_samples:
        print("\n详细列表:")
        print("-"*80)
        for sample in worsened_samples[:10]:  # 显示前10个
            print(f"\n样本 {sample['index']+1}:")
            print(f"  Ground Truth: {sample['ground_truth']}")
            print(f"  原始多数投票: {'✓ 正确' if sample['best_is_correct'] else '✗ 错误'}")
            print(f"  Geometric: {'✓ 正确' if sample['geo_correct'] else '✗ 错误'}")
            print(f"  Arithmetic: {'✓ 正确' if sample['arith_correct'] else '✗ 错误'}")
            
            # 显示前向和后向分数
            print(f"  前向分数: {sample['forward_scores']}")
            print(f"  后向分数: {sample['backward_scores']}")
            
            # 显示后向预测详情
            if sample['backward_details']:
                print("  后向预测详情:")
                for candidate, details in sample['backward_details'].items():
                    if candidate in sample['backward_scores']:
                        print(f"    {candidate}: 正确预测 {details.get('correct_count', 0)}/{details.get('total_numbers', 0)}")
    
    # ============ (2) 正确和错误rollout的数字预测准确率 ============
    print("\n" + "="*100)
    print("(2) 正确 vs 错误 Rollout 的数字预测准确率分析")
    print("="*100)
    
    # 收集所有rollout的数字预测信息
    rollout_analysis = []
    
    for r in results:
        if 'original_sample' in r and 'rollouts' in r['original_sample']:
            ground_truth = r['original_sample'].get('ground_truth_answer')
            
            for rollout_idx, rollout in enumerate(r['original_sample']['rollouts']):
                rollout_correct = rollout.get('is_correct', 0)
                extracted_answer = rollout.get('extracted_answer', '')
                
                # 从backward_details中获取这个rollout对应的数字预测信息
                backward_details = r.get('backward_details', {})
                
                # 找到这个rollout对应的候选答案
                candidate_info = backward_details.get(extracted_answer, {})
                predictions = candidate_info.get('predictions', [])
                
                # 统计这个rollout的数字预测情况
                correct_predictions = sum(1 for p in predictions if p.get('is_correct', False))
                total_predictions = len(predictions)
                
                rollout_analysis.append({
                    'sample_idx': r.get('index', 0),
                    'rollout_idx': rollout_idx,
                    'extracted_answer': extracted_answer,
                    'is_correct': rollout_correct,
                    'ground_truth': ground_truth,
                    'correct_predictions': correct_predictions,
                    'total_predictions': total_predictions,
                    'prediction_accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
                    'predictions_detail': predictions
                })
    
    # 按是否正确分组统计
    correct_rollouts = [r for r in rollout_analysis if r['is_correct'] == 1]
    wrong_rollouts = [r for r in rollout_analysis if r['is_correct'] == 0]
    
    print(f"\n总rollout数: {len(rollout_analysis)}")
    print(f"正确rollout数: {len(correct_rollouts)}")
    print(f"错误rollout数: {len(wrong_rollouts)}")
    
    if correct_rollouts:
        correct_pred_acc = np.mean([r['prediction_accuracy'] for r in correct_rollouts])
        correct_pred_std = np.std([r['prediction_accuracy'] for r in correct_rollouts])
        print(f"\n正确Rollout的数字预测准确率:")
        print(f"  均值: {correct_pred_acc:.4f} ({correct_pred_acc*100:.2f}%)")
        print(f"  标准差: {correct_pred_std:.4f}")
        print(f"  中位数: {np.median([r['prediction_accuracy'] for r in correct_rollouts]):.4f}")
    
    if wrong_rollouts:
        wrong_pred_acc = np.mean([r['prediction_accuracy'] for r in wrong_rollouts])
        wrong_pred_std = np.std([r['prediction_accuracy'] for r in wrong_rollouts])
        print(f"\n错误Rollout的数字预测准确率:")
        print(f"  均值: {wrong_pred_acc:.4f} ({wrong_pred_acc*100:.2f}%)")
        print(f"  标准差: {wrong_pred_std:.4f}")
        print(f"  中位数: {np.median([r['prediction_accuracy'] for r in wrong_rollouts]):.4f}")
    
    # 统计检验
    if correct_rollouts and wrong_rollouts:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            [r['prediction_accuracy'] for r in correct_rollouts],
            [r['prediction_accuracy'] for r in wrong_rollouts]
        )
        print(f"\n统计检验 (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        if p_value < 0.05:
            print(f"  ✓ 差异显著 (p < 0.05)")
        else:
            print(f"  ✗ 差异不显著 (p >= 0.05)")
    
    # ============ 额外分析：按答案类型分组 ============
    print("\n" + "="*100)
    print("(3) 按答案类型分组分析")
    print("="*100)
    
    # 按答案分组统计
    answer_stats = defaultdict(lambda: {'correct': [], 'wrong': [], 'pred_acc': []})
    
    for r in rollout_analysis:
        answer = r['extracted_answer']
        answer_stats[answer]['pred_acc'].append(r['prediction_accuracy'])
        if r['is_correct'] == 1:
            answer_stats[answer]['correct'].append(r)
        else:
            answer_stats[answer]['wrong'].append(r)
    
    print("\n各答案类型的统计:")
    print("-"*80)
    for answer, stats in sorted(answer_stats.items(), key=lambda x: len(x[1]['pred_acc']), reverse=True):
        total = len(stats['pred_acc'])
        correct_count = len(stats['correct'])
        wrong_count = len(stats['wrong'])
        avg_pred_acc = np.mean(stats['pred_acc']) if stats['pred_acc'] else 0
        
        print(f"\n答案: {answer}")
        print(f"  出现次数: {total}")
        print(f"  正确rollout数: {correct_count} ({correct_count/total*100:.1f}%)")
        print(f"  错误rollout数: {wrong_count} ({wrong_count/total*100:.1f}%)")
        print(f"  平均数字预测准确率: {avg_pred_acc:.4f} ({avg_pred_acc*100:.2f}%)")
    
    # ============ 保存详细分析结果 ============
    analysis_output = {
        'worsened_samples': worsened_samples,
        'rollout_analysis': rollout_analysis,
        'statistics': {
            'total_rollouts': len(rollout_analysis),
            'correct_rollouts': len(correct_rollouts),
            'wrong_rollouts': len(wrong_rollouts),
            'correct_rollouts_pred_acc': {
                'mean': np.mean([r['prediction_accuracy'] for r in correct_rollouts]) if correct_rollouts else None,
                'std': np.std([r['prediction_accuracy'] for r in correct_rollouts]) if correct_rollouts else None,
                'median': np.median([r['prediction_accuracy'] for r in correct_rollouts]) if correct_rollouts else None
            },
            'wrong_rollouts_pred_acc': {
                'mean': np.mean([r['prediction_accuracy'] for r in wrong_rollouts]) if wrong_rollouts else None,
                'std': np.std([r['prediction_accuracy'] for r in wrong_rollouts]) if wrong_rollouts else None,
                'median': np.median([r['prediction_accuracy'] for r in wrong_rollouts]) if wrong_rollouts else None
            }
        }
    }
    
    # 保存到文件
    with open('detailed_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_output, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*100)
    print(f"详细分析结果已保存到: detailed_analysis.json")
    print("="*100)
    
    return analysis_output


def analyze_correct_vs_wrong_predictions(results_file_path):
    """
    更细粒度的分析：每个数字的预测情况
    """
    with open(results_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "="*100)
    print("(4) 每个数字的预测准确率分析")
    print("="*100)
    
    # 收集每个数字的预测情况
    digit_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for r in results:
        backward_details = r.get('backward_details', {})
        
        for candidate, details in backward_details.items():
            predictions = details.get('predictions', [])
            for pred in predictions:
                original_num = pred.get('original_number', '')
                is_correct = pred.get('is_correct', False)
                
                if original_num:
                    digit_stats[original_num]['total'] += 1
                    if is_correct:
                        digit_stats[original_num]['correct'] += 1
    
    print("\n各数字的预测准确率:")
    print("-"*80)
    for digit, stats in sorted(digit_stats.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total']>0 else 0):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {digit:10s}: {stats['correct']:4d}/{stats['total']:4d} = {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FOBAR结果详细分析')
    parser.add_argument('results_file', type=str, help='FOBAR结果JSON文件路径')
    parser.add_argument('--digit-analysis', action='store_true', help='包含每个数字的详细分析')
    
    args = parser.parse_args()
    
    # 详细分析
    analysis = detailed_analysis(args.results_file)
    
    # 数字级别分析
    if args.digit_analysis:
        analyze_correct_vs_wrong_predictions(args.results_file)