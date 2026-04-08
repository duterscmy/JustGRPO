#!/usr/bin/env python3
"""
合并多个FOBAR评估结果文件

用法:
    python merge_results.py result1.json result2.json result3.json -o merged_result.json
    python merge_results.py results/*.json -o merged_result.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_result(filepath: str) -> Dict[str, Any]:
    """加载单个结果文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_summaries(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个文件的summary部分
    对于相同策略，累加correct和total，重新计算accuracy
    """
    merged_summary = {
        "strategies": {},
        "total_samples": 0
    }
    
    # 用于累加每个策略的数据
    strategy_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in results:
        summary = result.get("summary", {})
        strategies = summary.get("strategies", {})
        
        # 累加每个策略的数据
        for strategy_name, strategy_data in strategies.items():
            strategy_stats[strategy_name]["correct"] += strategy_data.get("correct", 0)
            strategy_stats[strategy_name]["total"] += strategy_data.get("total", 0)
        
        # 取最大的total_samples（所有文件应该相同）
        if summary.get("total_samples", 0) > merged_summary["total_samples"]:
            merged_summary["total_samples"] = summary.get("total_samples", 0)
    
    # 计算合并后的accuracy
    for strategy_name, stats in strategy_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total > 0 else 0.0
        
        merged_summary["strategies"][strategy_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    return merged_summary


def merge_detailed_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个文件的detailed_results部分
    按照样本顺序拼接所有文件的详细结果
    """
    merged_detailed = defaultdict(lambda: {
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "details": []
    })
    
    for result in results:
        detailed = result.get("detailed_results", {})
        
        for strategy_name, strategy_data in detailed.items():
            # 累加统计信息
            merged_detailed[strategy_name]["correct"] += strategy_data.get("correct", 0)
            merged_detailed[strategy_name]["total"] += strategy_data.get("total", 0)
            
            # 合并详细结果列表
            details_list = strategy_data.get("details", [])
            merged_detailed[strategy_name]["details"].extend(details_list)
    
    # 重新计算accuracy
    for strategy_name, data in merged_detailed.items():
        total = data["total"]
        correct = data["correct"]
        data["accuracy"] = correct / total if total > 0 else 0.0
    
    return dict(merged_detailed)


def merge_results(result_files: List[str], output_file: str, verbose: bool = True):
    """
    合并多个结果文件
    
    Args:
        result_files: 结果文件路径列表
        output_file: 输出文件路径
        verbose: 是否打印详细信息
    """
    if verbose:
        print(f"找到 {len(result_files)} 个结果文件")
        print("-" * 60)
    
    # 加载所有结果
    all_results = []
    for filepath in result_files:
        if verbose:
            print(f"加载: {filepath}")
        try:
            result = load_result(filepath)
            all_results.append(result)
        except Exception as e:
            print(f"警告: 无法加载 {filepath} - {e}")
    
    if not all_results:
        print("错误: 没有成功加载任何结果文件")
        return
    
    if verbose:
        print("-" * 60)
        print("合并中...")
    
    # 合并summary
    merged_summary = merge_summaries(all_results)
    
    # 合并detailed_results
    merged_detailed = merge_detailed_results(all_results)
    
    # 构建最终结果
    merged_result = {
        "summary": merged_summary,
        "detailed_results": merged_detailed,
        "merge_info": {
            "source_files": result_files,
            "num_sources": len(result_files),
            "total_samples_merged": merged_summary["total_samples"]
        }
    }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"✓ 结果已保存到: {output_file}")
        print("-" * 60)
        print("\n合并后的统计:")
        print(f"总样本数: {merged_summary['total_samples']}")
        print("\n各策略准确率:")
        for strategy_name, stats in merged_summary["strategies"].items():
            print(f"  {strategy_name}: {stats['correct']}/{stats['total']} = {stats['accuracy']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='合并多个FOBAR评估结果文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s result1.json result2.json result3.json -o merged.json
  %(prog)s results/*.json -o merged.json
  %(prog)s --input_list files.txt -o merged.json
        """
    )
    
    parser.add_argument('input_files', nargs='*', help='输入的结果文件路径（支持通配符）')
    parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    parser.add_argument('-l', '--input_list', help='包含输入文件路径的文本文件（每行一个路径）')
    parser.add_argument('-q', '--quiet', action='store_true', help='静默模式，不打印详细信息')
    
    args = parser.parse_args()
    
    # 收集所有输入文件
    input_files = list(args.input_files) if args.input_files else []
    
    # 如果提供了文件列表，读取文件内容
    if args.input_list:
        with open(args.input_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    input_files.append(line)
    
    if not input_files:
        print("错误: 请提供至少一个输入文件")
        parser.print_help()
        return
    
    # 去重并排序
    input_files = sorted(set(input_files))
    
    # 合并结果
    merge_results(input_files, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()