#!/usr/bin/env python3
"""
下载 HuggingFaceH4/MATH-500 数据集并保存为 CSV 文件
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset


def download_math500(save_path=None, split="test"):
    """
    下载 MATH-500 数据集并保存为 CSV
    
    Args:
        save_path: 保存CSV的路径，默认为当前目录下的 math500.csv
        split: 数据集划分，可选 'test' 或其他（该数据集主要包含test集）
    
    Returns:
        df: 返回加载的DataFrame
    """
    
    print(f"正在加载 MATH-500 数据集 ({split} 集)...")
    
    # 加载数据集
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
    
    # 转换为 pandas DataFrame
    df = dataset.to_pandas()
    
    print(f"数据集加载成功！包含 {len(df)} 个样本")
    print(f"数据列: {list(df.columns)}")
    
    # 设置保存路径
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "math500.csv")
    elif os.path.isdir(save_path):
        save_path = os.path.join(save_path, "math500.csv")
    
    # 保存为 CSV
    print(f"正在保存到: {save_path}")
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"保存完成！")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='下载 MATH-500 数据集并保存为 CSV')
    parser.add_argument('--save-path', '-o', type=str, default=None,
                        help='保存CSV的路径（默认: 当前目录/math500.csv）')
    parser.add_argument('--split', '-s', type=str, default='test',
                        help='数据集划分（默认: test）')
    
    args = parser.parse_args()
    
    try:
        df = download_math500(args.save_path, args.split)
        
        # 显示前几行数据预览
        print("\n数据预览（前3行）:")
        print(df.head(3))
        
        # 显示基本信息
        print(f"\n数据集信息:")
        print(f"- 总样本数: {len(df)}")
        print(f"- 列名: {', '.join(df.columns)}")
        print(f"- 内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
