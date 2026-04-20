import json
import re
import argparse
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
from tqdm import tqdm
import torch
import traceback
import time
from transformers import AutoTokenizer, AutoModel


class LLaDADiffusionLM:
    """封装LLaDA模型，提供反向预测接口"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        print(f"Loading LLaDA model from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        
        self.mask_token_id = 126336
        self.mask_token_text = self.tokenizer.decode([self.mask_token_id])
        print(f"Model loaded. Mask token: '{self.mask_token_text}'")
    
    def predict_masked(self, text: str, max_retries: int = 3) -> Tuple[str, List[str]]:
        """预测文本中所有[MASK]位置的token，带重试机制"""
        for attempt in range(max_retries):
            try:
                # 限制输入长度，避免过长
                if len(text) > 4000:
                    print(f"  ⚠️  Input too long ({len(text)} chars), truncating...")
                    text = text[:4000]
                
                encoded = self.tokenizer(
                    [text],
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt"
                )
                input_ids = encoded['input_ids'].to(self.device)
                
                # 检查输入长度
                if input_ids.shape[1] > 2048:
                    print(f"  ⚠️  Token sequence too long ({input_ids.shape[1]} tokens), truncating...")
                    input_ids = input_ids[:, :2048]
                
                with torch.no_grad():
                    logits = self.model(input_ids).logits
                    mask_positions = (input_ids == self.mask_token_id)
                    
                    if not mask_positions.any():
                        return text, []
                    
                    predicted_token_ids = torch.argmax(logits, dim=-1)
                    predicted_tokens = []
                    for pos in mask_positions[0].nonzero(as_tuple=True)[0].tolist():
                        token_id = predicted_token_ids[0, pos].item()
                        token_text = self.tokenizer.decode([token_id])
                        predicted_tokens.append(token_text)
                    
                    output_ids = input_ids.clone()
                    output_ids[mask_positions] = predicted_token_ids[mask_positions]
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                return output_text, predicted_tokens
                
            except Exception as e:
                print(f"  ⚠️  Prediction attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    return text, []
                time.sleep(1)
        
        return text, []


class BackwardVerifier:
    """反向验证器：对每个候选答案，验证其能否正确预测被mask的数字"""
    
    def __init__(self, diffusion_lm: LLaDADiffusionLM):
        self.diffusion_lm = diffusion_lm
    
    def _extract_numbers(self, text: str) -> List[Tuple[str, Tuple[int, int], str]]:
        """从文本中提取所有数字及其位置"""
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
    
    def _mask_number(self, text: str, positions: Tuple[int, int]) -> str:
        """用[MASK]替换指定位置的数字"""
        start, end = positions
        original_num = text[start:end]
        try:
            num_tokens = self.diffusion_lm.tokenizer.encode(original_num, add_special_tokens=False)
        except Exception as e:
            print(f"    ⚠️  Tokenization failed for '{original_num}': {e}")
            num_tokens = [self.diffusion_lm.tokenizer.unk_token_id] if self.diffusion_lm.tokenizer.unk_token_id else [0]
        mask_text = self.diffusion_lm.mask_token_text * max(1, len(num_tokens))
        return text[:start] + mask_text + text[end:]
    
    def verify_candidate(self, user: str, assistant: str, candidate: str, 
                         verbose: bool = False) -> Dict[str, Any]:
        """
        验证单个候选答案
        """
        numbers = self._extract_numbers(user)
        
        if verbose:
            print(f"    Numbers found: {[n[0] for n in numbers]}")
        
        if not numbers:
            return {
                "candidate": candidate,
                "total_numbers": 0,
                "correct_count": 0,
                "backward_score": 0.5,
                "digit_results": [],
                "numbers": [],
                "error": "No numbers in question"
            }
        
        digit_results = []
        correct_count = 0
        
        for idx, (original_num, positions, context) in enumerate(numbers):
            if verbose:
                print(f"      Processing number {idx+1}/{len(numbers)}: '{original_num}'")
            
            try:
                # 构造masked问题
                masked_user = self._mask_number(user, positions)
                # 反向输入（限制assistant长度）
                assistant_short = assistant[:1000] if len(assistant) > 1000 else assistant
                backward_input = f"{masked_user}\n\n{assistant_short}"
                
                if verbose:
                    print(f"        Backward input length: {len(backward_input)} chars")
                
                # 预测
                _, predicted_tokens = self.diffusion_lm.predict_masked(backward_input)
                predicted_num = ''.join(predicted_tokens).strip() if predicted_tokens else ""
                
                if verbose:
                    print(f"        Original: '{original_num}' → Predicted: '{predicted_num}'")
                
                is_correct = (predicted_num == original_num)
                if is_correct:
                    correct_count += 1
                
                digit_results.append({
                    "original": original_num,
                    "predicted": predicted_num,
                    "is_correct": is_correct,
                    "context": context
                })
                
            except Exception as e:
                print(f"      ❌ Error processing number {idx+1}: {e}")
                digit_results.append({
                    "original": original_num,
                    "predicted": "",
                    "is_correct": False,
                    "context": context,
                    "error": str(e)
                })
        
        total = len(numbers)
        return {
            "candidate": candidate,
            "total_numbers": total,
            "correct_count": correct_count,
            "backward_score": correct_count / total if total > 0 else 0.5,
            "digit_results": digit_results,
            "numbers": [{"value": n[0], "context": n[2]} for n in numbers],
        }


def parse_ground_truth(text: str) -> Tuple[str, str]:
    """从文本中提取最终答案"""
    if not text:
        return "", ""
    
    # 尝试提取 \\boxed{} 中的内容
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return text, boxed_match.group(1).strip()
    
    # 尝试提取 "The answer is X" 格式
    answer_match = re.search(r'(?:The|the)\s+answer\s+is\s+([A-D]|\d+(?:\.\d+)?)', text)
    if answer_match:
        return text, answer_match.group(1).strip()
    
    # 取最后一行或默认
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        num_match = re.search(r'(\d+(?:\.\d+)?)', last_line)
        if num_match:
            return text, num_match.group(1).strip()
    
    return text, ""


def load_dataset(input_path: str) -> List[Dict]:
    """加载数据集"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {input_path}")
    return data


def save_dataset(data: List[Dict], output_path: str):
    """保存数据集"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute backward verification results')
    parser.add_argument('input_file', type=str, help='Input JSON file path')
    parser.add_argument('output_file', type=str, help='Output JSON file path')
    parser.add_argument('--model_path', type=str,
                        default='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',
                        help='Path to LLaDA model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Limit number of samples for testing')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed debug info')
    args = parser.parse_args()
    
    # 加载模型
    diffusion_lm = LLaDADiffusionLM(args.model_path, args.device)
    verifier = BackwardVerifier(diffusion_lm)
    
    # 加载数据
    dataset = load_dataset(args.input_file)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")
    
    # 为每个样本添加 backward_result
    failed_samples = []
    
    for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            user = sample.get('question', sample.get('prompt', ''))
            rollouts = sample.get('rollouts', [])
            
            if not user:
                print(f"\n⚠️  Sample {sample_idx}: No question/prompt field")
                sample['backward_result'] = []
                sample['forward_votes'] = {}
                continue
            
            if not rollouts:
                print(f"\n⚠️  Sample {sample_idx}: No rollouts")
                sample['backward_result'] = []
                sample['forward_votes'] = {}
                continue
            
            # 提取所有候选答案
            # candidates = []
            # candidate_to_assistant = {}
            # for rollout_idx, rollout in enumerate(rollouts):
            #     try:
            #         _, answer = parse_ground_truth(rollout)
            #         if answer and answer not in candidate_to_assistant:
            #             candidate_to_assistant[answer] = rollout
            #             candidates.append(answer)
            #     except Exception as e:
            #         print(f"\n⚠️  Sample {sample_idx}, rollout {rollout_idx}: Parse error: {e}")
            
            # if args.verbose:
            #     print(f"\n📊 Sample {sample_idx}: Found {len(candidates)} candidates: {candidates}")
            
            # 对每个候选答案执行反向验证
            backward_results = []
            for rollout in rollouts:
                _, answer = parse_ground_truth(rollout)
                result = verifier.verify_candidate(user, rollout, answer, verbose=args.verbose)
                backward_results.append(result)
            
            # 添加到样本
            sample['backward_result'] = backward_results
            
        except Exception as e:
            print(f"\n❌ Sample {sample_idx} failed: {e}")
            traceback.print_exc()
            failed_samples.append(sample_idx)
            sample['backward_result'] = []
            # sample['forward_votes'] = {}
            sample['backward_error'] = str(e)
    
    # 保存
    save_dataset(dataset, args.output_file)
    
    # 打印统计
    print("\n" + "="*60)
    print("Backward Results Summary")
    print("="*60)
    
    success_count = 0
    for sample_idx, sample in enumerate(dataset[:10]):  # 只显示前10个
        if 'backward_error' not in sample:
            success_count += 1
            print(f"\nSample {sample_idx}:")
            for br in sample.get('backward_result', []):
                print(f"  Candidate: {br['candidate']} -> score: {br['backward_score']:.3f} ({br.get('correct_count', 0)}/{br.get('total_numbers', 0)})")
    
    if len(dataset) > 10:
        print(f"\n... and {len(dataset)-10} more samples")
    
    print(f"\n✅ Successfully processed: {success_count}/{len(dataset)} samples")
    if failed_samples:
        print(f"❌ Failed samples: {failed_samples}")
    
    # 额外打印统计信息
    total_candidates = sum(len(s.get('backward_result', [])) for s in dataset)
    print(f"Total candidates verified: {total_candidates}")


if __name__ == "__main__":
    main()