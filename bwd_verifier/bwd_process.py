import torch
import re
import json
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


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


class FOBARWithLLaDA:
    """使用LLaDA作为Diffusion LM的FOBAR实现"""
    
    def __init__(self, diffusion_lm=None, verbose=True):
        self.diffusion_lm = diffusion_lm
        self.verbose = verbose
    
    def extract_numbers(self, text: str) -> List[Tuple[str, Tuple[int, int], str]]:
        """
        提取文本中的所有数字，返回更详细的信息
        返回: [(数字字符串, (起始位置, 结束位置), 原始上下文片段), ...]
        """
        pattern = r'-?\d+\.?\d*'
        numbers = []
        for match in re.finditer(pattern, text):
            num_str = match.group()
            if num_str and num_str not in ['-', '.']:
                # 获取上下文（前后各10个字符）
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                context = text[start:end]
                numbers.append((num_str, (match.start(), match.end()), context))
        return numbers
    
    def mask_single_number(self, text: str, number_info: Tuple[str, Tuple[int, int], str]) -> str:
        """
        将单个数字替换为对应token数量的[MASK]
        需要先tokenize来准确知道该数字占多少个token
        """
        num_str, (start, end), _ = number_info
        
        # 获取数字在文本中的原始字符串
        original_num = text[start:end]
        
        # Tokenize这个数字，看它占多少个token
        num_tokens = self.diffusion_lm.tokenizer.encode(original_num, add_special_tokens=False)
        num_token_count = len(num_tokens)
        
        # 生成对应数量的[MASK]
        masks = self.diffusion_lm.mask_token_text * num_token_count
        
        return text[:start] + masks + text[end:]
    
    def compute_forward_scores(self, rollouts: List[Dict]) -> Dict[str, float]:
        """计算前向分数"""
        answers = [r["extracted_answer"] for r in rollouts]
        counter = Counter(answers)
        total = len(rollouts)
        return {ans: count/total for ans, count in counter.items()}
    
    def compute_backward_scores(self, 
                                user: str, 
                                candidate_answers: Set[str],
                                rollouts: List[Dict]) -> Tuple[Dict[str, float], Dict]:
        """
        使用LLaDA计算反向分数，同时返回详细的预测信息
        """
        if self.diffusion_lm is None:
            print("Warning: No diffusion_lm provided")
            return {ans: 0.0 for ans in candidate_answers}, {}
        
        # 提取user中的所有数字
        numbers = self.extract_numbers(user)
        if not numbers:
            print(f"Warning: No numbers found in user question")
            return {ans: 0.0 for ans in candidate_answers}, {}
        else:
            print(f"Extracted numbers from user question: {[num for num, _, _ in numbers]}")
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Found {len(numbers)} numbers in user question:")
            for i, (num, _, context) in enumerate(numbers):
                print(f"  [{i+1}] {num} (context: ...{context}...)")
            print(f"{'='*80}")
        
        backward_scores = {}
        detailed_info = {}  # 存储每个候选答案的详细信息
        
        for candidate in candidate_answers:
            print(f"\nProcessing candidate answer: {candidate}")
            # 找到对应的assistant
            assistant = None
            assistant_idx = None
            for idx, r in enumerate(rollouts):
                if r["extracted_answer"] == candidate:
                    assistant = r["assistant"]
                    assistant_idx = idx
                    break
            
            if not assistant:
                if self.verbose:
                    print(f"\n  ⚠️  Warning: No assistant for candidate {candidate}")
                backward_scores[candidate] = 0.0
                detailed_info[candidate] = {
                    "assistant_index": None,
                    "predictions": [],
                    "correct_count": 0,
                    "total_numbers": len(numbers)
                }
                continue
            
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"📊 Processing candidate: {candidate} (from rollout {assistant_idx})")
                print(f"{'─'*80}")
            
            correct_count = 0
            predictions = []
            
            for i, (original_num, positions, context) in enumerate(numbers):
                # 创建masked user
                masked_user = self.mask_single_number(user, (original_num, positions, context))
                
                # 构建反向输入
                backward_input = f"{masked_user}\n\n{assistant}"
                
                if self.verbose:
                    print(f"\n  🔍 Testing number {i+1}/{len(numbers)}: {original_num}")
                    print(f"     Masked input preview: {masked_user}")
                
                # LLaDA预测
                predicted_text, pred_info = self.diffusion_lm.predict_masked(backward_input)
                print(f"     Predicted text preview: {predicted_text[:len(masked_user)]}")
                # 从预测文本中提取数字
                predicted_num = ''.join(pred_info["predicted_tokens"]).strip()
                
                # 记录详细预测信息
                predictions.append({
                    "original_number": original_num,
                    "predicted_number": predicted_num,
                    "is_correct": 1 if predicted_num==original_num else False,
                    "context": context,
                    "masked_position": positions,
                    "prediction_info": {
                        "mask_positions": pred_info['mask_positions'],
                        "predicted_tokens": pred_info['predicted_tokens'],
                        "num_masks": pred_info['num_masks']
                    }
                })

                if predicted_num == original_num:
                    correct_count += 1
            
            score = correct_count / len(numbers)
            backward_scores[candidate] = score
            
            detailed_info[candidate] = {
                "assistant_index": assistant_idx,
                "assistant_preview": assistant[:200] + "..." if len(assistant) > 200 else assistant,
                "predictions": predictions,
                "correct_count": correct_count,
                "total_numbers": len(numbers),
                "score": score
            }
            
            if self.verbose:
                print(f"\n  📈 Candidate {candidate} summary:")
                print(f"     Correct: {correct_count}/{len(numbers)} = {score:.3f}")
                print(f"{'─'*80}")
        
        return backward_scores, detailed_info
    
    def combine_scores(self, 
                      forward_scores: Dict[str, float], 
                      backward_scores: Dict[str, float],
                      method: str = "geometric") -> Dict[str, float]:
        """组合分数"""
        combined = {}
        all_answers = set(forward_scores.keys()) | set(backward_scores.keys())
        
        for ans in all_answers:
            f = forward_scores.get(ans, 0.0)
            b = backward_scores.get(ans, 0.0)
            
            if method == "geometric":
                combined[ans] = np.sqrt(f * b) if f > 0 and b > 0 else 0.0
            else:  # arithmetic
                combined[ans] = (f + b) / 2
        
        return combined
    
    def process_sample(self, sample: Dict, methods: List[str] = ["geometric", "arithmetic"]) -> Dict:
        """处理单个样本"""
        rollouts = sample["rollouts"]
        ground_truth = sample["ground_truth_answer"]
        user = rollouts[0]["user"]
        
        if self.verbose:
            print(f"\n{'#'*80}")
            print(f"Processing new sample")
            print(f"Ground truth answer: {ground_truth}")
            print(f"Number of rollouts: {len(rollouts)}")
            print(f"{'#'*80}")
        
        # 前向分数
        forward_scores = self.compute_forward_scores(rollouts)
        candidate_answers = set(forward_scores.keys())
        
        if self.verbose:
            print(f"\n📊 Forward scores:")
            for ans, score in sorted(forward_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"   {ans}: {score:.3f}")
        
        # 反向分数
        backward_scores, backward_details = self.compute_backward_scores(user, candidate_answers, rollouts)
        
        # 组合
        results = {}
        for method in methods:
            combined_scores = self.combine_scores(forward_scores, backward_scores, method)
            best_answer = max(combined_scores, key=combined_scores.get)
            best_score = combined_scores[best_answer]
            
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"📌 {method.upper()} results:")
                for ans, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
                    marker = "✓" if ans == str(ground_truth) else ""
                    print(f"   {ans}: {score:.3f} {marker}")
                print(f"   Selected: {best_answer} (score: {best_score:.3f})")
                print(f"   Correct: {best_answer == str(ground_truth)}")
            
            results[method] = {
                "selected_answer": best_answer,
                "selected_score": best_score,
                "combined_scores": combined_scores,
                "is_correct": (best_answer == str(ground_truth))
            }
        
        return {
            "original_sample": sample,
            "fobar_results": results,
            "forward_scores": forward_scores,
            "backward_scores": backward_scores,
            "backward_details": backward_details,  # 添加详细信息
            "ground_truth": ground_truth
        }


def process_dataset(dataset_path: str, output_path: str, model_path: str, device='cuda', verbose=True):
    """处理整个数据集"""
    # 加载数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 初始化LLaDA
    print("Loading LLaDA model...")
    diffusion_lm = LLaDADiffusionLM(model_path=model_path, device=device)
    
    # 初始化FOBAR
    fobar = FOBARWithLLaDA(diffusion_lm, verbose=verbose)
    
    # 处理每个样本
    results = []
    for i, sample in enumerate(dataset):
        print(f"\n{'='*80}")
        print(f"Processing sample {i+1}/{len(dataset)}")
        print(f"{'='*80}")
        
        result = fobar.process_sample(sample)
        results.append(result)
    
    # 统计
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    
    for method in ["geometric", "arithmetic"]:
        correct = sum(1 for r in results if r["fobar_results"][method]["is_correct"])
        total = len(results)
        print(f"\n{method.upper()} accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FOBAR with LLaDA - 使用扩散语言模型进行后向验证')
    parser.add_argument('input_json', type=str, help='输入JSON文件路径')
    parser.add_argument('output_json', type=str, help='输出JSON文件路径')
    parser.add_argument('--model','-m', type=str, default='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', help='LLaDA模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true', default=True, help='显示详细信息')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='不显示详细信息')
    
    args = parser.parse_args()
    
    results = process_dataset(
        args.input_json,
        args.output_json,
        args.model,
        device=args.device,
        verbose=args.verbose
    )