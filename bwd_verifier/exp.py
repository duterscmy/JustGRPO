import json
import argparse
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import sys, os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grader import math_equal
from utils.parser import extract_answer, parse_ground_truth

class LLaDADiffusionLM:
    """
    封装LLaDA模型，提供predict_masked接口
    用于FOBAR的后向推理
    """
    
    def __init__(self, model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', device='cuda'):
        """
        初始化LLaDA模型和tokenizer
        """
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # LLaDA使用左填充
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        
        # mask token ID
        self.mask_token_id = 126336
        
        print(f"LLaDA model loaded on {device}")
        print(f"Mask token ID: {self.mask_token_id}")       
        print(f"Mask token text: '{self.tokenizer.decode([self.mask_token_id])}'")
        self.mask_token_text = self.tokenizer.decode([self.mask_token_id])
    def predict_masked(self, masked_text: str, temperature: float = 0.0) -> Tuple[str, dict]:
        """
        预测masked文本中被mask的数字
        
        Returns:
            (predicted_text, info) 其中info包含预测的详细信息
        """
        # 1. Tokenize输入
        # messages = [{"role": "user", "content": masked_text}]
        # prompt = self.tokenizer.apply_chat_template(
        #     messages, 
        #     add_generation_prompt=True, 
        #     tokenize=False
        # )
        
        # 编码
        encoded = self.tokenizer(
            [masked_text],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 2. 单次forward预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 找到mask token的位置
            mask_positions = (input_ids == self.mask_token_id)
            
            if not mask_positions.any():
                return masked_text, {"mask_positions": [], "predicted_tokens": []}
            
            # 3. 预测
            if temperature == 0:
                predicted_token_ids = torch.argmax(logits, dim=-1)
            else:
                logits_with_noise = self._add_gumbel_noise(logits, temperature)
                predicted_token_ids = torch.argmax(logits_with_noise, dim=-1)
            
            # 4. 只替换mask位置
            output_ids = input_ids.clone()
            output_ids[mask_positions] = predicted_token_ids[mask_positions]
            
            # 5. 解码
            predicted_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 6. 记录预测信息
            mask_positions_list = mask_positions[0].nonzero(as_tuple=True)[0].tolist()
            predicted_tokens = []
            for pos in mask_positions_list:
                token_id = predicted_token_ids[0, pos].item()
                token_text = self.tokenizer.decode([token_id])
                predicted_tokens.append(token_text)
            
            info = {
                "mask_positions": mask_positions_list,
                "predicted_tokens": predicted_tokens,
                "num_masks": len(mask_positions_list)
            }
            print(info)
            
        return predicted_text, info
    
    def _add_gumbel_noise(self, logits, temperature):
        """添加Gumbel噪声用于采样"""
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise



class AnswerSelector:
    """答案选择器，支持多种策略"""
    
    def __init__(self, strategy='first', diffusion_lm=None, device='cuda'):
        """
        Args:
            strategy: 选择策略
                - 'first': 直接选择第一个rollout
                - 'majority': 多数投票
                - 'fobar': 前向+后向验证（需要diffusion_lm）
            diffusion_lm: LLaDA模型实例（仅fobar策略需要）
            device: 设备
        """
        self.strategy = strategy
        self.diffusion_lm = diffusion_lm
        self.device = device
    
    def select_answer(self, sample: Dict) -> Tuple[str, Dict]:
        """根据策略选择最佳答案"""
        rollouts = sample['rollouts']
        
        if self.strategy == 'first':
            return self._select_first(rollouts)
        elif self.strategy == 'majority':
            return self._select_majority(rollouts)
        elif self.strategy == 'fobar':
            return self._select_fobar(sample)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _select_first(self, rollouts: List[str]) -> Tuple[str, Dict]:
        """直接选择第一个rollout的答案"""
        # parse_ground_truth 返回 (something, answer) 元组
        first_answer = parse_ground_truth(rollouts[0])[1]
        return first_answer, {
            "strategy": "first",
            "selected_index": 0,
            "all_answers": [parse_ground_truth(r)[1] for r in rollouts]
        }
    
    def _select_majority(self, rollouts: List[str]) -> Tuple[str, Dict]:
        """多数投票选择答案"""
        answers = [parse_ground_truth(r)[1] for r in rollouts]
        
        # 统计每个答案出现的次数
        answer_counts = Counter(answers)
        
        # 找出最高频的答案
        most_common = answer_counts.most_common(1)[0]
        selected_answer = most_common[0]
        max_count = most_common[1]
        
        return selected_answer, {
            "strategy": "majority",
            "answer_counts": dict(answer_counts),
            "selected_count": max_count,
            "total_rollouts": len(rollouts),
            "all_answers": answers
        }
    
    def _select_fobar(self, sample: Dict) -> Tuple[str, Dict]:
        """
        使用FOBAR策略选择答案
        需要实现后向验证
        """
        if self.diffusion_lm is None:
            print("Warning: No diffusion_lm provided for fobar strategy, falling back to majority")
            return self._select_majority(sample['rollouts'])
        
        rollouts = sample['rollouts']
        user = sample.get('question', sample.get('prompt', ''))
        
        # 1. 提取所有答案
        answers = [parse_ground_truth(r)[1] for r in rollouts]
        unique_answers = list(set(answers))
        
        # 2. 计算前向分数（频率）
        answer_counts = Counter(answers)
        forward_scores = {ans: count/len(rollouts) for ans, count in answer_counts.items()}
        
        # 3. 计算后向分数（使用LLaDA验证）
        backward_scores = self._compute_backward_scores(user, unique_answers, rollouts)
        
        # 4. 组合分数（几何平均）
        combined_scores = {}
        for ans in unique_answers:
            f = forward_scores.get(ans, 0.0)
            b = backward_scores.get(ans, 0.0)
            combined_scores[ans] = np.sqrt(f * b) if f > 0 and b > 0 else 0.0
        
        # 5. 选择分数最高的答案
        selected_answer = max(combined_scores, key=combined_scores.get)
        
        return selected_answer, {
            "strategy": "fobar",
            "forward_scores": forward_scores,
            "backward_scores": backward_scores,
            "combined_scores": combined_scores,
            "all_answers": answers
        }
    
    def _compute_backward_scores(self, user: str, candidate_answers: List[str], 
                                      rollouts: List[str]) -> Dict[str, float]:
        """
        计算反向分数（适配Mask Predict模型）
        
        关键修改：
        1. Zc: 统计所有mask位置中预测正确的总数（不是比例）
        2. PB: 归一化概率，而非正确率
        3. 使用mask token ID而非字符串"[MASK]"
        """
        # 提取user中的所有数字
        numbers = self._extract_numbers(user)
        
        if not numbers:
            # 没有数字时均匀分配概率
            n_candidates = len(candidate_answers)
            return {ans: 1.0/n_candidates for ans in candidate_answers}
        
        Z_dict = {}  # 存储每个候选答案的Zc（正确预测总数）
        
        for candidate in candidate_answers:
            # 找到对应的assistant response
            assistant = None
            for r in rollouts:
                if parse_ground_truth(r)[1] == candidate:
                    assistant = r
                    break
            
            if not assistant:
                Z_dict[candidate] = 0
                continue
            
            Zc = 0  # 论文公式(2): 正确预测的总次数
            
            # 对每个被mask的数字位置
            for original_num, positions, context in numbers:
                # 创建masked question（使用真正的mask token）
                masked_user = self._mask_single_number_with_token(user, positions)
                
                # 构建反向输入（不需要template，因为mask已经暗示了要预测什么）
                backward_input = f"{masked_user}\n\n{assistant}"
                
                # LLaDA预测（单次，因为是mask predict）
                predicted_text, pred_info = self.diffusion_lm.predict_masked(backward_input)
                
                # 提取预测的数字
                predicted_num = self._extract_predicted_number(pred_info, original_num)
                
                # 论文公式(2): 累加正确预测
                if predicted_num == original_num:
                    Zc += 1
            
            Z_dict[candidate] = Zc
        
        # 论文公式(3): 计算PB（归一化概率）
        total_Z = sum(Z_dict.values())
        epsilon = 1e-8  # 避免除零
        PB = {}
        
        for candidate, Zc in Z_dict.items():
            # PB(Aˆc) = (Zc + ε) / (sum(Zc') + ε*|A|)
            PB[candidate] = (Zc + epsilon) / (total_Z + epsilon * len(candidate_answers))
        
        return PB

    def _mask_single_number_with_token(self, text: str, positions: Tuple[int, int]) -> str:
        """
        使用真正的mask token替换数字（而非字符串"[MASK]"）
        需要tokenize来精确定位
        """
        start, end = positions
        original_num = text[start:end]
        
        # Tokenize原始数字，看它占多少个token
        num_tokens = self.diffusion_lm.tokenizer.encode(original_num, add_special_tokens=False)
        num_token_count = len(num_tokens)
        
        # 生成对应数量的[MASK] token
        mask_token_text = self.diffusion_lm.tokenizer.decode([self.diffusion_lm.mask_token_id])
        masks = mask_token_text * num_token_count
        
        return text[:start] + masks + text[end:]

    def _extract_predicted_number(self, pred_info: dict, original_num: str) -> str:
        """
        从预测信息中提取数字
        处理可能的格式差异
        """
        # 方法1: 从predicted_tokens拼接
        predicted_tokens = pred_info.get("predicted_tokens", [])
        predicted_str = ''.join(predicted_tokens).strip()
        
        # 清理可能的特殊token
        predicted_str = re.sub(r'<\|.*?\|>', '', predicted_str)
        
        # 如果预测结果为空或无效，返回空字符串
        if not predicted_str:
            return ""
        
        # 尝试提取数字（可能预测结果包含多余字符）
        import re
        numbers = re.findall(r'-?\d+\.?\d*', predicted_str)
        
        if numbers:
            return numbers[0]  # 返回第一个数字
        
        return predicted_str


def evaluate_single_sample(args_tuple):
    """
    评估单个样本（用于多进程）
    注意：这个函数需要在模块级别定义，以便可以被pickle序列化
    """
    sample, strategy, ground_truth_key, diffusion_lm = args_tuple
    
    # 创建选择器
    selector = AnswerSelector(strategy=strategy, diffusion_lm=diffusion_lm)
    
    # 选择答案
    selected_answer, info = selector.select_answer(sample)
    
    # 获取标准答案
    ground_truth = sample.get(ground_truth_key, '')
    
    # 判断是否正确
    is_correct = math_equal(selected_answer, ground_truth)
    
    return {
        "strategy": strategy,
        "selected_answer": selected_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "details": info
    }


class ParallelEvaluator:
    """并行评估器，支持多进程"""
    
    def __init__(self, strategies: List[str] = ['first', 'majority'], 
                 diffusion_lm=None, num_workers: int = None,
                 ground_truth_key: str = 'answer'):
        """
        Args:
            strategies: 要评估的策略列表
            diffusion_lm: LLaDA模型实例（仅fobar策略需要）
            num_workers: 并行进程数，默认为CPU核心数
            ground_truth_key: ground truth在sample中的key
        """
        self.strategies = strategies
        self.diffusion_lm = diffusion_lm
        self.num_workers = num_workers or cpu_count()
        self.ground_truth_key = ground_truth_key
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """并行评估整个数据集"""
        results = {}
        
        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy}")
            print(f"Using {self.num_workers} workers")
            print(f"{'='*60}")
            
            # 准备参数
            args_list = [
                (sample, strategy, self.ground_truth_key, self.diffusion_lm if strategy == 'fobar' else None)
                for sample in dataset
            ]
            
            # 使用多进程池并行处理
            sample_results = []
            
            with Pool(processes=self.num_workers) as pool:
                # 使用tqdm显示进度
                for result in tqdm(
                    pool.imap_unordered(evaluate_single_sample, args_list),
                    total=len(dataset),
                    desc=f"Processing {strategy}",
                    unit="sample"
                ):
                    sample_results.append(result)
            
            # 统计结果
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
    """顺序评估器（原始版本，用于对比）"""
    
    def __init__(self, strategies: List[str] = ['first', 'majority'], 
                 diffusion_lm=None, ground_truth_key: str = 'answer'):
        self.strategies = strategies
        self.diffusion_lm = diffusion_lm
        self.ground_truth_key = ground_truth_key
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """顺序评估整个数据集"""
        results = {}
        
        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy}")
            print(f"{'='*60}")
            
            correct_count = 0
            sample_results = []
            selector = AnswerSelector(strategy=strategy, 
                                     diffusion_lm=self.diffusion_lm if strategy == 'fobar' else None)
            
            for i, sample in enumerate(tqdm(dataset, desc=f"Processing {strategy}", unit="sample")):
                # 选择答案
                selected_answer, info = selector.select_answer(sample)
                
                # 获取标准答案
                ground_truth = sample.get(self.ground_truth_key, '')
                
                # 判断是否正确
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
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def save_results(results: Dict, output_path: str):
    """保存评估结果"""
    # 提取总结信息
    summary = {
        "strategies": {},
        "total_samples": 0
    }
    
    for strategy, data in results.items():
        summary["strategies"][strategy] = {
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"]
        }
        if not summary["total_samples"]:
            summary["total_samples"] = data["total"]
    
    # 保存完整结果
    output_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # 打印总结
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for strategy, data in summary["strategies"].items():
        print(f"{strategy.upper()}: {data['correct']}/{data['total']} = {data['accuracy']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate different answer selection strategies')
    parser.add_argument('input_file', type=str, help='Input JSON file path')
    parser.add_argument('output_file', type=str, help='Output JSON file path')
    parser.add_argument('--strategies', type=str, nargs='+', 
                        default=['first', 'majority'],
                        choices=['first', 'majority', 'fobar'],
                        help='Strategies to evaluate')
    parser.add_argument('--use_fobar', action='store_true', 
                        help='Enable FOBAR strategy (requires LLaDA model)')
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',
                        help='Path to LLaDA model for FOBAR')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for LLaDA model')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    # parser.add_argument('--sequential', action='store_true',
    #                     help='Use sequential processing instead of parallel')
    
    args = parser.parse_args()
    
    # 加载数据集
    print(f"Loading dataset from {args.input_file}")
    dataset = load_dataset(args.input_file)
    print(f"Loaded {len(dataset)} samples")
    
    # 初始化LLaDA（如果需要FOBAR策略）
    diffusion_lm = None
    if args.use_fobar or 'fobar' in args.strategies:
        print("\nLoading LLaDA model for FOBAR strategy...")
        try:
            diffusion_lm = LLaDADiffusionLM(
                model_path=args.model_path,
                device=args.device
            )
            print("LLaDA model loaded successfully")
        except ImportError:
            print("Warning: Could not import LLaDADiffusionLM, FOBAR will not work")
            if 'fobar' in args.strategies:
                args.strategies.remove('fobar')
                print("Removed fobar from strategies")
    
    # 选择评估器
    if 'fobar' in args.strategies:
        print("\nUsing sequential evaluation...")
        evaluator = SequentialEvaluator(
            strategies=args.strategies,
            diffusion_lm=diffusion_lm
        )
    else:
        print(f"\nUsing parallel evaluation with {args.num_workers or cpu_count()} workers...")
        evaluator = ParallelEvaluator(
            strategies=args.strategies,
            diffusion_lm=diffusion_lm,
            num_workers=args.num_workers
        )
    
    # 评估
    results = evaluator.evaluate_dataset(dataset)
    
    # 保存结果
    save_results(results, args.output_file)


if __name__ == "__main__":
    import re  # 添加缺失的导入
    main()