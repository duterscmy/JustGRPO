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
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grader import math_equal
from utils.parser import extract_answer, parse_ground_truth

# 全局计时器
_timers = {}

def timer_start(name):
    """开始计时"""
    _timers[name] = time.time()

def timer_end(name, threshold=0.01):
    """结束计时并打印（超过阈值才打印）"""
    if name in _timers:
        elapsed = time.time() - _timers[name]
        if elapsed > threshold:
            print(f"⏱️  [{name}] took {elapsed:.3f}s")
        return elapsed
    return 0

class LLaDADiffusionLM:
    """
    封装LLaDA模型，提供predict_masked接口
    用于FOBAR的后向推理
    """
    
    def __init__(self, model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', device='cuda'):
        """
        初始化LLaDA模型和tokenizer
        """
        timer_start("LLaDA_init")
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
        timer_end("LLaDA_init", threshold=0.1)
        
    def predict_masked(self, masked_text: str, temperature: float = 0.0) -> Tuple[str, dict]:
        """预测masked文本中被mask的数字"""
        timer_start("predict_masked")
        
        # 1. Tokenize输入
        messages = [{"role": "user", "content": masked_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
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
                timer_end("predict_masked", threshold=0.01)
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
            
        timer_end("predict_masked", threshold=0.01)
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
        self._cache = {}  # 添加缓存
        self._stats = {
            'select_calls': 0,
            'total_time': 0,
            'parse_time': 0,
            'confidence_time': 0
        }

    def select_answer(self, sample: Dict) -> Tuple[str, Dict]:
        """根据策略选择最佳答案"""
        timer_start(f"select_{self.strategy}")
        
        cache_key = (
            sample.get('question', sample.get('prompt', '')),
            self.strategy
        )
        
        if cache_key in self._cache:
            timer_end(f"select_{self.strategy}", threshold=0.01)
            return self._cache[cache_key]
        
        rollouts = sample['rollouts']
        rollouts_records = sample.get('rollouts_records', [])  # 获取置信度记录
        
        # 记录数据大小
        num_rollouts = len(rollouts)
        num_records = len(rollouts_records) if rollouts_records else 0
        
        result = None
        if self.strategy == 'first':
            result = self._select_first(rollouts)
        elif self.strategy == 'majority':
            result = self._select_majority(rollouts)
        elif self.strategy == 'highest_confidence':
            result = self._select_highest_confidence(rollouts, rollouts_records)
        elif self.strategy == 'weighted_confidence':
            result = self._select_weighted_confidence(rollouts, rollouts_records)
        elif self.strategy == 'confidence_threshold':
            threshold = sample.get('confidence_threshold', 0.9)
            result = self._select_confidence_threshold(rollouts, rollouts_records, threshold)
        elif self.strategy == 'fobar':
            result = self._select_fobar(sample)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # 缓存结果
        self._cache[cache_key] = result
        
        elapsed = timer_end(f"select_{self.strategy}", threshold=0.05)
        self._stats['select_calls'] += 1
        self._stats['total_time'] += elapsed
        
        # 每100个样本打印一次统计
        if self._stats['select_calls'] % 100 == 0:
            avg_time = self._stats['total_time'] / self._stats['select_calls']
            print(f"📊 [{self.strategy}] Avg select time: {avg_time:.3f}s (calls: {self._stats['select_calls']})")
        
        return result
    
    def _select_first(self, rollouts: List[str]) -> Tuple[str, Dict]:
        """直接选择第一个rollout的答案"""
        timer_start("parse_first")
        first_answer = parse_ground_truth(rollouts[0])[1]
        all_answers = [parse_ground_truth(r)[1] for r in rollouts]
        timer_end("parse_first", threshold=0.01)
        
        return first_answer, {
            "strategy": "first",
            "selected_index": 0,
            "all_answers": all_answers
        }
    
    def _select_majority(self, rollouts: List[str]) -> Tuple[str, Dict]:
        """多数投票选择答案"""
        timer_start("parse_majority")
        answers = [parse_ground_truth(r)[1] for r in rollouts]
        timer_end("parse_majority", threshold=0.01)
        
        timer_start("count_majority")
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        selected_answer = most_common[0]
        max_count = most_common[1]
        timer_end("count_majority", threshold=0.01)
        
        return selected_answer, {
            "strategy": "majority",
            "answer_counts": dict(answer_counts),
            "selected_count": max_count,
            "total_rollouts": len(rollouts),
            "all_answers": answers
        }
    
    def _select_fobar(self, sample: Dict) -> Tuple[str, Dict]:
        """使用FOBAR策略选择答案"""
        if self.diffusion_lm is None:
            print("Warning: No diffusion_lm provided for fobar strategy, falling back to majority")
            return self._select_majority(sample['rollouts'])
        
        rollouts = sample['rollouts']
        user = sample.get('question', sample.get('prompt', ''))
        
        # 1. 提取所有答案
        timer_start("fobar_parse")
        answers = [parse_ground_truth(r)[1] for r in rollouts]
        unique_answers = list(set(answers))
        timer_end("fobar_parse", threshold=0.01)
        
        # 2. 计算前向分数
        timer_start("fobar_forward")
        answer_counts = Counter(answers)
        forward_scores = {ans: count/len(rollouts) for ans, count in answer_counts.items()}
        timer_end("fobar_forward", threshold=0.01)
        
        # 3. 计算后向分数
        timer_start("fobar_backward")
        backward_scores = self._compute_backward_scores(user, unique_answers, rollouts)
        timer_end("fobar_backward", threshold=0.05)
        
        # 4. 组合分数
        timer_start("fobar_combine")
        combined_scores = {}
        for ans in unique_answers:
            f = forward_scores.get(ans, 0.0)
            b = backward_scores.get(ans, 0.0)
            combined_scores[ans] = np.sqrt(f * b) if f > 0 and b > 0 else 0.0
        
        selected_answer = max(combined_scores, key=combined_scores.get)
        timer_end("fobar_combine", threshold=0.01)
        
        return selected_answer, {
            "strategy": "fobar",
            "forward_scores": forward_scores,
            "backward_scores": backward_scores,
            "combined_scores": combined_scores,
            "all_answers": answers
        }
    
    def _compute_backward_scores(self, user: str, candidate_answers: List[str], 
                                      rollouts: List[str]) -> Dict[str, float]:
        """计算反向分数"""
        # 提取user中的所有数字
        timer_start("extract_numbers")
        numbers = self._extract_numbers(user)
        timer_end("extract_numbers", threshold=0.01)
        
        if not numbers:
            n_candidates = len(candidate_answers)
            return {ans: 1.0/n_candidates for ans in candidate_answers}
        
        Z_dict = {}
        
        for candidate in candidate_answers:
            timer_start(f"backward_candidate_{candidate[:20]}")
            
            # 找到对应的assistant response
            assistant = None
            for r in rollouts:
                if parse_ground_truth(r)[1] == candidate:
                    assistant = r
                    break
            
            if not assistant:
                Z_dict[candidate] = 0
                timer_end(f"backward_candidate_{candidate[:20]}", threshold=0.01)
                continue
            
            Zc = 0
            
            # 对每个被mask的数字位置
            for original_num, positions, context in numbers:
                # 创建masked question
                masked_user = self._mask_single_number_with_token(user, positions)
                backward_input = f"{masked_user}\n\n{assistant}"
                
                # LLaDA预测
                predicted_text, pred_info = self.diffusion_lm.predict_masked(backward_input)
                
                # 提取预测的数字
                predicted_num = self._extract_predicted_number(pred_info, original_num)
                
                if predicted_num == original_num:
                    Zc += 1
            
            Z_dict[candidate] = Zc
            timer_end(f"backward_candidate_{candidate[:20]}", threshold=0.05)
        
        # 计算PB
        timer_start("normalize_scores")
        total_Z = sum(Z_dict.values())
        epsilon = 1e-8
        PB = {}
        
        for candidate, Zc in Z_dict.items():
            PB[candidate] = (Zc + epsilon) / (total_Z + epsilon * len(candidate_answers))
        timer_end("normalize_scores", threshold=0.01)
        
        return PB

    def _mask_single_number_with_token(self, text: str, positions: Tuple[int, int]) -> str:
        """使用真正的mask token替换数字"""
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
        """从预测信息中提取数字"""
        predicted_tokens = pred_info.get("predicted_tokens", [])
        predicted_str = ''.join(predicted_tokens).strip()
        predicted_str = re.sub(r'<\|.*?\|>', '', predicted_str)
        
        if not predicted_str:
            return ""
        
        numbers = re.findall(r'-?\d+\.?\d*', predicted_str)
        if numbers:
            return numbers[0]
        return predicted_str

    def _extract_numbers(self, text: str) -> List[Tuple[str, Tuple[int, int], str]]:
        """从文本中提取数字"""
        pattern = r'-?\d+\.?\d*'
        numbers = []
        for match in re.finditer(pattern, text):
            num_str = match.group()
            if num_str and num_str not in ['-', '.']:
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                context = text[start:end]
                numbers.append((num_str, (match.start(), match.end()), context))
        return numbers

    def _select_highest_confidence(self, rollouts: List[str], rollouts_records: List[List[Dict]]) -> Tuple[str, Dict]:
        """选择置信度最高的rollout的答案"""
        if not rollouts_records or len(rollouts_records) != len(rollouts):
            print("Warning: No confidence records found, falling back to first rollout")
            return self._select_first(rollouts)
        
        timer_start("confidence_extract")
        confidence_scores = []
        
        for idx, records in enumerate(rollouts_records):
            # 直接提取置信度（已经排好序）
            token_confidences = [record.get('confidence', 0.0) for record in records]
            
            if token_confidences:
                avg_confidence = sum(token_confidences) / len(token_confidences)
                score = avg_confidence
                
                confidence_scores.append({
                    "rollout_idx": idx,
                    "score": score,
                    "avg_confidence": avg_confidence,
                    "num_tokens": len(token_confidences),
                    "token_confidences": token_confidences
                })
            else:
                confidence_scores.append({
                    "rollout_idx": idx,
                    "score": 0.0,
                    "avg_confidence": 0.0,
                    "num_tokens": 0,
                    "token_confidences": []
                })
        timer_end("confidence_extract", threshold=0.01)
        
        timer_start("confidence_select")
        best_rollout = max(confidence_scores, key=lambda x: x['score'])
        best_idx = best_rollout['rollout_idx']
        
        timer_start("confidence_parse_answer")
        selected_answer = parse_ground_truth(rollouts[best_idx])[1]
        all_answers = [parse_ground_truth(r)[1] for r in rollouts]
        timer_end("confidence_parse_answer", threshold=0.01)
        timer_end("confidence_select", threshold=0.01)
        
        return selected_answer, {
            "strategy": "highest_confidence",
            "selected_index": best_idx,
            "selected_confidence": best_rollout['score'],
            "selected_confidence_details": {
                "avg": best_rollout['avg_confidence'],
                "num_tokens": best_rollout['num_tokens']
            },
            "all_confidence_scores": [
                {
                    "rollout_idx": cs['rollout_idx'],
                    "score": cs['score'],
                    "avg_confidence": cs['avg_confidence'],
                    "num_tokens": cs['num_tokens']
                }
                for cs in confidence_scores
            ],
            "all_answers": all_answers
        }

    def _select_weighted_confidence(self, rollouts: List[str], rollouts_records: List[List[Dict]]) -> Tuple[str, Dict]:
        """基于置信度加权投票选择答案"""
        if not rollouts_records or len(rollouts_records) != len(rollouts):
            print("Warning: No confidence records found, falling back to majority")
            return self._select_majority(rollouts)
        
        timer_start("weighted_extract")
        answer_confidence = {}
        answer_votes = {}
        rollout_details = []
        
        for idx, records in enumerate(rollouts_records):
            # 提取答案
            answer = parse_ground_truth(rollouts[idx])[1]
            
            # 提取置信度
            token_confidences = [record.get('confidence', 0.0) for record in records]
            
            if token_confidences:
                weight = sum(token_confidences) / len(token_confidences)
            else:
                weight = 0.0
            
            if answer not in answer_confidence:
                answer_confidence[answer] = 0.0
                answer_votes[answer] = 0
            answer_confidence[answer] += weight
            answer_votes[answer] += 1
            
            rollout_details.append({
                "rollout_idx": idx,
                "answer": answer,
                "confidence_weight": weight,
                "num_tokens": len(token_confidences)
            })
        timer_end("weighted_extract", threshold=0.01)
        
        timer_start("weighted_select")
        selected_answer = max(answer_confidence, key=answer_confidence.get)
        
        all_answers = [parse_ground_truth(r)[1] for r in rollouts]
        answer_counts = Counter(all_answers)
        forward_scores = {ans: count/len(rollouts) for ans, count in answer_counts.items()}
        timer_end("weighted_select", threshold=0.01)
        
        return selected_answer, {
            "strategy": "weighted_confidence",
            "selected_answer": selected_answer,
            "selected_confidence": answer_confidence[selected_answer],
            "answer_confidence_scores": answer_confidence,
            "answer_votes": answer_votes,
            "forward_scores": forward_scores,
            "rollout_details": rollout_details,
            "all_answers": all_answers
        }

    def _select_confidence_threshold(self, rollouts: List[str], rollouts_records: List[List[Dict]], 
                                    threshold: float = 0.9) -> Tuple[str, Dict]:
        """基于置信度阈值选择"""
        if not rollouts_records or len(rollouts_records) != len(rollouts):
            print("Warning: No confidence records found, falling back to majority")
            return self._select_majority(rollouts)
        
        timer_start("threshold_extract")
        high_confidence_answers = []
        rollout_filter_details = []
        
        for idx, records in enumerate(rollouts_records):
            token_confidences = [record.get('confidence', 0.0) for record in records]
            
            if token_confidences:
                avg_confidence = sum(token_confidences) / len(token_confidences)
            else:
                avg_confidence = 0.0
            
            answer = parse_ground_truth(rollouts[idx])[1]
            
            rollout_filter_details.append({
                "rollout_idx": idx,
                "answer": answer,
                "avg_confidence": avg_confidence,
                "passed_threshold": avg_confidence >= threshold
            })
            
            if avg_confidence >= threshold:
                high_confidence_answers.append(answer)
        timer_end("threshold_extract", threshold=0.01)
        
        if not high_confidence_answers:
            print(f"Warning: No rollout meets confidence threshold {threshold}, falling back to majority")
            return self._select_majority(rollouts)
        
        timer_start("threshold_vote")
        answer_counts = Counter(high_confidence_answers)
        most_common = answer_counts.most_common(1)[0]
        selected_answer = most_common[0]
        max_count = most_common[1]
        
        all_answers = [parse_ground_truth(r)[1] for r in rollouts]
        timer_end("threshold_vote", threshold=0.01)
        
        return selected_answer, {
            "strategy": "confidence_threshold",
            "threshold": threshold,
            "selected_answer": selected_answer,
            "selected_count": max_count,
            "total_high_confidence_rollouts": len(high_confidence_answers),
            "answer_counts": dict(answer_counts),
            "rollout_filter_details": rollout_filter_details,
            "all_answers": all_answers
        }


def evaluate_single_sample(args_tuple):
    """
    评估单个样本（用于多进程）
    注意：这个函数需要在模块级别定义，以便可以被pickle序列化
    """
    sample, strategy, ground_truth_key, diffusion_lm, sample_idx = args_tuple
    
    timer_start(f"sample_{sample_idx}_{strategy}")
    
    # 记录样本基本信息
    num_rollouts = len(sample.get('rollouts', []))
    has_records = 'rollouts_records' in sample and sample['rollouts_records']
    
    if sample_idx % 50 == 0:  # 每50个样本打印一次
        print(f"🔍 Processing sample {sample_idx} (rollouts: {num_rollouts}, has_records: {has_records})")
    
    # 创建选择器
    timer_start(f"create_selector_{sample_idx}")
    selector = AnswerSelector(strategy=strategy, diffusion_lm=diffusion_lm)
    timer_end(f"create_selector_{sample_idx}", threshold=0.001)
    
    # 选择答案
    timer_start(f"select_answer_{sample_idx}")
    try:
        selected_answer, info = selector.select_answer(sample)
    except Exception as e:
        print(f"❌ Error in sample {sample_idx}: {e}")
        traceback.print_exc()
        selected_answer = ""
        info = {"error": str(e)}
    elapsed_select = timer_end(f"select_answer_{sample_idx}", threshold=0.05)
    
    # 获取标准答案
    ground_truth = sample.get(ground_truth_key, '')
    
    # 判断是否正确
    timer_start(f"math_equal_{sample_idx}")
    is_correct = math_equal(selected_answer, ground_truth)
    elapsed_math = timer_end(f"math_equal_{sample_idx}", threshold=0.01)
    
    total_elapsed = timer_end(f"sample_{sample_idx}_{strategy}", threshold=0.1)
    
    # 打印慢样本
    if total_elapsed > 1.0:  # 超过1秒
        print(f"🐌 Slow sample {sample_idx}: total={total_elapsed:.2f}s, select={elapsed_select:.2f}s, math={elapsed_math:.4f}s, rollouts={num_rollouts}")
    
    return {
        "sample_idx": sample_idx,
        "strategy": strategy,
        "selected_answer": selected_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "details": info
    }


class ParallelEvaluator:
    """并行评估器，支持多进程（保持结果有序）"""
    
    def __init__(self, strategies: List[str] = ['first', 'majority'], 
                 diffusion_lm=None, num_workers: int = None,
                 ground_truth_key: str = 'answer'):
        self.strategies = strategies
        self.diffusion_lm = diffusion_lm
        self.num_workers = num_workers or cpu_count()
        self.ground_truth_key = ground_truth_key
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """并行评估整个数据集（保持原始顺序）"""
        results = {}
        
        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy}")
            print(f"Using {self.num_workers} workers")
            print(f"Dataset size: {len(dataset)}")
            print(f"{'='*60}")
            
            # 检查数据集是否有rollouts_records
            has_records = any('rollouts_records' in s and s['rollouts_records'] for s in dataset)
            print(f"📊 Dataset has confidence records: {has_records}")
            
            # 准备参数
            args_list = [
                (sample, strategy, self.ground_truth_key, 
                 self.diffusion_lm if strategy == 'fobar' else None, 
                 idx)
                for idx, sample in enumerate(dataset)
            ]
            
            start_time = time.time()
            
            with Pool(processes=self.num_workers) as pool:
                unordered_results = []
                
                # 使用imap_unordered并显示进度
                for result in tqdm(
                    pool.imap_unordered(evaluate_single_sample, args_list),
                    total=len(dataset),
                    desc=f"Processing {strategy}",
                    unit="sample",
                    ncols=80
                ):
                    unordered_results.append(result)
            
            elapsed = time.time() - start_time
            speed = len(dataset) / elapsed
            print(f"\n⏱️  Strategy {strategy} completed in {elapsed:.1f}s ({speed:.2f} samples/sec)")
            
            # 按原始顺序排序
            unordered_results.sort(key=lambda x: x['sample_idx'])
            for result in unordered_results:
                del result['sample_idx']
            
            sample_results = unordered_results
            
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
    """顺序评估器（用于对比）"""
    
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
            
            start_time = time.time()
            correct_count = 0
            sample_results = []
            selector = AnswerSelector(strategy=strategy, 
                                     diffusion_lm=self.diffusion_lm if strategy == 'fobar' else None)
            
            for i, sample in enumerate(tqdm(dataset, desc=f"Processing {strategy}", unit="sample")):
                selected_answer, info = selector.select_answer(sample)
                ground_truth = sample.get(self.ground_truth_key, '')
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
            
            elapsed = time.time() - start_time
            speed = len(dataset) / elapsed
            print(f"\n⏱️  Strategy {strategy} completed in {elapsed:.1f}s ({speed:.2f} samples/sec)")
            
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
    print(f"Loading dataset from {filepath}")
    start_time = time.time()
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    elapsed = time.time() - start_time
    print(f"Loaded {len(dataset)} samples in {elapsed:.2f}s")
    
    # 检查第一个样本的结构
    if dataset:
        sample0 = dataset[0]
        print(f"\n📋 Sample structure:")
        print(f"  Keys: {list(sample0.keys())}")
        if 'rollouts' in sample0:
            print(f"  Number of rollouts: {len(sample0['rollouts'])}")
        if 'rollouts_records' in sample0:
            records = sample0['rollouts_records']
            print(f"  Has rollouts_records: {bool(records)}")
            if records and len(records) > 0:
                print(f"  First record length: {len(records[0]) if records[0] else 0}")
    
    return dataset


def save_results(results: Dict, output_path: str):
    """保存评估结果"""
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
    
    output_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for strategy, data in summary["strategies"].items():
        print(f"{strategy.upper()}: {data['correct']}/{data['total']} = {data['accuracy']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate different answer selection strategies')
    parser.add_argument('input_file', type=str, help='Input JSON file path')
    parser.add_argument('output_file', type=str, help='Output JSON file path')
    parser.add_argument('--strategies', '-s', type=str, nargs='+', 
                        default=['first', 'majority', 'highest_confidence', 'weighted_confidence', 'confidence_threshold'],
                        choices=['first', 'majority', 'fobar','highest_confidence', 'weighted_confidence', 'confidence_threshold'],
                        help='Strategies to evaluate')
    parser.add_argument('--use_fobar', action='store_true', 
                        help='Enable FOBAR strategy (requires LLaDA model)')
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',
                        help='Path to LLaDA model for FOBAR')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for LLaDA model')
    parser.add_argument('--num_workers','-n', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    # 加载数据集
    dataset = load_dataset(args.input_file)
    
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
        except Exception as e:
            print(f"Warning: Could not load LLaDA model: {e}")
            if 'fobar' in args.strategies:
                args.strategies.remove('fobar')
                print("Removed fobar from strategies")
    
    # 选择评估器
    if 'fobar' in args.strategies:
        print("\n⚠️  FOBAR strategy detected, using sequential evaluation (parallel not supported for FOBAR)")
        evaluator = SequentialEvaluator(
            strategies=args.strategies,
            diffusion_lm=diffusion_lm
        )
    else:
        print(f"\n✅ Using parallel evaluation with {args.num_workers or cpu_count()} workers...")
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
    main()