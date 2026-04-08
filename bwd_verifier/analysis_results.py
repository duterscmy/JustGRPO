#!/usr/bin/env python3
"""
分析两个方法的评估结果，找出方法1正确但方法2错误的样例

用法:
    python analysis_results.py merged.json fobar majority output.json
    python analysis_results.py merged.json first majority output.json
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def load_merged_results(filepath: str) -> Dict[str, Any]:
    """加载合并后的结果文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_sample_results(detailed_results: Dict, method1: str, method2: str) -> List[Tuple[int, Dict]]:
    """
    提取两个方法在相同样本上的结果
    
    Returns:
        List of (sample_index, sample_data) where sample_data contains both methods' results
    """
    # 获取两个方法的详细结果列表
    method1_details = detailed_results.get(method1, {}).get("details", [])
    method2_details = detailed_results.get(method2, {}).get("details", [])
    
    # 确保两个列表长度相同
    if len(method1_details) != len(method2_details):
        print(f"警告: {method1}有{len(method1_details)}个样本，{method2}有{len(method2_details)}个样本")
        min_len = min(len(method1_details), len(method2_details))
        method1_details = method1_details[:min_len]
        method2_details = method2_details[:min_len]
    
    # 配对样本
    paired_results = []
    for idx, (sample1, sample2) in enumerate(zip(method1_details, method2_details)):
        # 验证样本的基本信息一致
        if sample1.get("ground_truth") != sample2.get("ground_truth"):
            print(f"警告: 样本{idx}的ground_truth不一致: {sample1.get('ground_truth')} vs {sample2.get('ground_truth')}")
            continue
        
        paired_results.append((idx, {
            "ground_truth": sample1.get("ground_truth"),
            method1: {
                "selected_answer": sample1.get("selected_answer"),
                "is_correct": sample1.get("is_correct", False),
                "details": sample1.get("details", {})
            },
            method2: {
                "selected_answer": sample2.get("selected_answer"),
                "is_correct": sample2.get("is_correct", False),
                "details": sample2.get("details", {})
            }
        }))
    
    return paired_results


def find_method1_correct_method2_wrong(paired_results: List[Tuple[int, Dict]], 
                                        method1: str, method2: str) -> List[Dict]:
    """
    找出方法1正确但方法2错误的样例
    """
    wrong_cases = []
    
    for idx, data in paired_results:
        method1_correct = data[method1]["is_correct"]
        method2_correct = data[method2]["is_correct"]
        
        if method1_correct and not method2_correct:
            wrong_cases.append({
                "sample_index": idx,
                "ground_truth": data["ground_truth"],
                "method1_selected": data[method1]["selected_answer"],
                "method2_selected": data[method2]["selected_answer"],
                "method1_details": data[method1].get("details", {}),
                "method2_details": data[method2].get("details", {})
            })
    
    return wrong_cases


def analyze_wrong_cases(wrong_cases: List[Dict]) -> Dict[str, Any]:
    """
    分析错误样例的统计信息
    """
    if not wrong_cases:
        return {
            "total_wrong_cases": 0,
            "analysis": "No wrong cases found! Method1 is always correct when Method2 is wrong.",
            "common_wrong_answers": {},
            "method2_answer_distribution": {}
        }
    
    # 统计方法2最常见的错误答案
    method2_answers = [case["method2_selected"] for case in wrong_cases]
    method2_answer_counts = defaultdict(int)
    for ans in method2_answers:
        method2_answer_counts[ans] += 1
    
    # 按频率排序
    sorted_answers = sorted(method2_answer_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 统计ground_truth分布
    gt_distribution = defaultdict(int)
    for case in wrong_cases:
        gt_distribution[case["ground_truth"]] += 1
    
    return {
        "total_wrong_cases": len(wrong_cases),
        "percentage": len(wrong_cases) / len(method2_answers) * 100 if method2_answers else 0,
        "method2_answer_distribution": dict(sorted_answers[:10]),  # Top 10错误答案
        "ground_truth_distribution": dict(sorted(gt_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
        "most_common_wrong_answer": sorted_answers[0][0] if sorted_answers else None,
        "most_common_wrong_answer_count": sorted_answers[0][1] if sorted_answers else 0
    }


def save_analysis(output_file: str, method1: str, method2: str, 
                  wrong_cases: List[Dict], analysis: Dict[str, Any],
                  summary: Dict[str, Any]):
    """
    保存分析结果到JSON文件
    """
    output_data = {
        "analysis_config": {
            "method1": method1,
            "method2": method2,
            "description": f"Samples where {method1} is correct but {method2} is wrong"
        },
        "summary_statistics": {
            "total_samples": summary.get("total_samples", 0),
            "method1_correct": summary.get("method1_correct", 0),
            "method2_correct": summary.get("method2_correct", 0),
            "method1_accuracy": summary.get("method1_accuracy", 0),
            "method2_accuracy": summary.get("method2_accuracy", 0),
            "method1_correct_method2_wrong": len(wrong_cases),
            "method1_correct_method2_wrong_percentage": (len(wrong_cases) / summary.get("total_samples", 1)) * 100
        },
        "wrong_cases_analysis": analysis,
        "wrong_cases_details": wrong_cases
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 分析结果已保存到: {output_file}")


def print_analysis_report(method1: str, method2: str, 
                          wrong_cases: List[Dict], 
                          analysis: Dict[str, Any],
                          summary: Dict[str, Any]):
    """
    打印分析报告
    """
    print("\n" + "="*80)
    print(f"分析报告: {method1} 正确 但 {method2} 错误的样例")
    print("="*80)
    
    print(f"\n📊 总体统计:")
    print(f"  总样本数: {summary.get('total_samples', 0)}")
    print(f"  {method1} 准确率: {summary.get('method1_accuracy', 0):.2f}%")
    print(f"  {method2} 准确率: {summary.get('method2_accuracy', 0):.2f}%")
    print(f"  {method1} 正确但 {method2} 错误: {len(wrong_cases)} 个样本 ({analysis['percentage']:.2f}%)")
    
    if wrong_cases:
        print(f"\n🔍 错误分析:")
        print(f"  最常见的错误答案: '{analysis['most_common_wrong_answer']}' ({analysis['most_common_wrong_answer_count']}次, {analysis['most_common_wrong_answer_count']/len(wrong_cases)*100:.1f}%)")
        
        print(f"\n📈 方法2错误答案分布 (Top 10):")
        for ans, count in list(analysis['method2_answer_distribution'].items())[:10]:
            percentage = count / len(wrong_cases) * 100
            print(f"    '{ans}': {count}次 ({percentage:.1f}%)")
        
        print(f"\n🎯 Ground Truth分布 (Top 10):")
        for gt, count in list(analysis['ground_truth_distribution'].items())[:10]:
            percentage = count / len(wrong_cases) * 100
            print(f"    '{gt}': {count}次 ({percentage:.1f}%)")
        
        print(f"\n📝 前10个错误样例:")
        for i, case in enumerate(wrong_cases[:10]):
            print(f"\n  样例 {i+1} (索引 {case['sample_index']}):")
            print(f"    Ground Truth: {case['ground_truth']}")
            print(f"    {method1} 选择: {case['method1_selected']} ✓")
            print(f"    {method2} 选择: {case['method2_selected']} ✗")
    else:
        print(f"\n✨ 没有发现 {method1} 正确但 {method2} 错误的样例!")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='分析方法1正确但方法2错误的样例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s merged.json fobar majority output.json
  %(prog)s merged.json first majority output.json --verbose
        """
    )
    
    parser.add_argument('input_file', type=str, help='合并后的结果文件 (merged.json)')
    parser.add_argument('method1', type=str, help='正确的方法名称')
    parser.add_argument('method2', type=str, help='错误的方法名称')
    parser.add_argument('output_file', type=str, help='输出文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='打印详细信息')
    
    args = parser.parse_args()
    
    # 加载合并结果
    print(f"加载结果文件: {args.input_file}")
    merged_result = load_merged_results(args.input_file)
    
    # 提取摘要信息
    summary = merged_result.get("summary", {})
    strategies_summary = summary.get("strategies", {})
    detailed_results = merged_result.get("detailed_results", {})
    
    # 获取总体统计
    total_samples = summary.get("total_samples", 0)
    method1_stats = strategies_summary.get(args.method1, {})
    method2_stats = strategies_summary.get(args.method2, {})
    
    summary_info = {
        "total_samples": total_samples,
        "method1_correct": method1_stats.get("correct", 0),
        "method2_correct": method2_stats.get("correct", 0),
        "method1_accuracy": method1_stats.get("accuracy", 0) * 100,
        "method2_accuracy": method2_stats.get("accuracy", 0) * 100
    }
    
    # 提取配对的样本结果
    print(f"提取样本结果...")
    paired_results = extract_sample_results(detailed_results, args.method1, args.method2)
    print(f"找到 {len(paired_results)} 个配对样本")
    
    # 找出方法1正确但方法2错误的样例
    print(f"分析 {args.method1} 正确但 {args.method2} 错误的样例...")
    wrong_cases = find_method1_correct_method2_wrong(paired_results, args.method1, args.method2)
    
    # 分析错误样例
    analysis = analyze_wrong_cases(wrong_cases)
    
    # 打印报告
    print_analysis_report(args.method1, args.method2, wrong_cases, analysis, summary_info)
    
    # 保存结果
    save_analysis(args.output_file, args.method1, args.method2, wrong_cases, analysis, summary_info)
    
    # 如果verbose，打印更多细节
    if args.verbose and wrong_cases:
        print("\n" + "="*80)
        print("详细错误样例 (前20个):")
        print("="*80)
        for i, case in enumerate(wrong_cases[:20]):
            print(f"\n[{i+1}] 样本索引: {case['sample_index']}")
            print(f"    Ground Truth: {case['ground_truth']}")
            print(f"    {args.method1}: {case['method1_selected']}")
            print(f"    {args.method2}: {case['method2_selected']}")
            
            # 打印方法2的详细信息（如果有）
            if case['method2_details']:
                scores = case['method2_details'].get('combined_scores', {})
                if scores:
                    print(f"    {args.method2} 分数: {scores}")


if __name__ == "__main__":
    main()