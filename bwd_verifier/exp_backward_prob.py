import json
import re
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
import traceback
from transformers import AutoTokenizer, AutoModel


QUESTION_WORDS = ['how', 'what', 'when', 'where', 'who', 'which', 'why', 'whose', 'whom']


class LLaDADiffusionLM:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        print(f"Loading LLaDA model from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        self.mask_token_id = 126336
        self.mask_token_text = self.tokenizer.decode([self.mask_token_id])
        print(f"Model loaded. Mask token: '{self.mask_token_text}'")


def extract_key_span(question: str, max_words: int = 10) -> Tuple[str, str, str]:
    """
    提取关键片段，返回 (prefix, key_span, suffix)，三者拼接还原原始问题。

    逻辑：
      1. 找第一个 how/what/when/where/... 词（大小写不敏感）
      2. 从该词开始，按自然词切分，取最多 max_words 个词
      3. 找不到问题词则取最后 max_words 个词
    """
    pattern = r'\b(' + '|'.join(QUESTION_WORDS) + r')\b'
    match = re.search(pattern, question, re.IGNORECASE)

    if match:
        start_char = match.start()
    else:
        # fallback：取最后 max_words 个词
        words = question.split()
        if len(words) <= max_words:
            return '', question, ''
        fallback_span = ' '.join(words[-max_words:])
        start_char = question.rfind(fallback_span)
        if start_char == -1:
            return '', question, ''

    # 从 start_char 开始按空格切词，取最多 max_words 个
    remaining = question[start_char:]
    words = remaining.split()
    selected = words[:max_words]

    # 字符级定位 key_span 的结束位置
    pos = 0
    for w in selected:
        idx = remaining.find(w, pos)
        if idx == -1:
            break
        pos = idx + len(w)

    end_char = start_char + pos
    prefix = question[:start_char]
    key_span = question[start_char:end_char]
    suffix = question[end_char:]
    return prefix, key_span, suffix


class BackwardProbabilityVerifier:
    """
    只 mask 问题里的关键片段（how/what/... 开始的最多N个词），
    给定推理链，计算模型对该片段每个 token 的预测概率均值。
    """

    def __init__(self, diffusion_lm: LLaDADiffusionLM,
                 max_words: int = 10,
                 max_assistant_tokens: int = 512):
        self.lm = diffusion_lm
        self.max_words = max_words
        self.max_assistant_tokens = max_assistant_tokens

    def compute_probability_score(self, question: str, assistant: str,
                                   verbose: bool = False) -> Dict[str, Any]:
        try:
            # 1. 提取关键片段
            prefix, key_span, suffix = extract_key_span(question, self.max_words)

            if verbose:
                print(f"    prefix:   '{prefix}'")
                print(f"    key_span: '{key_span}'")
                print(f"    suffix:   '{suffix}'")

            # 2. tokenize 各部分
            prefix_ids  = self.lm.tokenizer.encode(prefix,    add_special_tokens=False) if prefix  else []
            key_ids     = self.lm.tokenizer.encode(key_span,  add_special_tokens=False)
            suffix_ids  = self.lm.tokenizer.encode(suffix,    add_special_tokens=False) if suffix  else []
            sep_ids     = self.lm.tokenizer.encode('\n\n',    add_special_tokens=False)
            asst_ids    = self.lm.tokenizer.encode(assistant, add_special_tokens=False)[:self.max_assistant_tokens]

            key_start = len(prefix_ids)
            key_len   = len(key_ids)

            # 3. 构造输入：prefix + [MASK]*key_len + suffix + \n\n + assistant
            input_ids = torch.tensor([[
                *prefix_ids,
                *([self.lm.mask_token_id] * key_len),
                *suffix_ids,
                *sep_ids,
                *asst_ids,
            ]], dtype=torch.long, device=self.lm.device)

            # 4. 前向推理
            with torch.no_grad():
                logits = self.lm.model(input_ids).logits  # (1, seq, vocab)

            # 5. 只取 key_span 位置的 logits → softmax → 正确 token 的概率
            key_logits = logits[0, key_start:key_start + key_len, :]
            probs = F.softmax(key_logits.float(), dim=-1)
            key_token_ids = torch.tensor(key_ids, device=self.lm.device)
            correct_probs = probs[
                torch.arange(key_len, device=self.lm.device), key_token_ids
            ].cpu().tolist()

            score = sum(correct_probs) / len(correct_probs) if correct_probs else 0.5
            key_token_texts = [self.lm.tokenizer.decode([tid]) for tid in key_ids]

            if verbose:
                print(f"    key tokens: {key_token_texts}")
                print(f"    probs:      {[f'{p:.3f}' for p in correct_probs]}")
                print(f"    score:      {score:.4f}")

            return {
                "backward_score_probability": score,
                "key_span": key_span,
                "key_token_count": key_len,
                "key_token_probs": correct_probs,
                "key_token_texts": key_token_texts,
                "found_question_word": bool(
                    re.search(r'\b(' + '|'.join(QUESTION_WORDS) + r')\b',
                              question, re.IGNORECASE)),
            }

        except Exception as e:
            traceback.print_exc()
            return {"backward_score_probability": 0.5, "error": str(e)}

    def verify_rollouts(self, question: str, rollouts: List[str],
                        verbose: bool = False) -> List[Dict[str, Any]]:
        results = []
        for idx, rollout in enumerate(rollouts):
            if verbose:
                print(f"\n  Rollout {idx + 1}/{len(rollouts)}")
            result = self.compute_probability_score(question, rollout, verbose=verbose)
            result["rollout_idx"] = idx
            results.append(result)
        return results


# ── 辅助 ─────────────────────────────────────────

def extract_user_question(sample: Dict) -> str:
    """优先取 question 字段，否则从 prompt 里解析用户消息"""
    q = sample.get('question', '')
    if q:
        return q
    prompt = sample.get('prompt', '')
    m = re.search(
        r'<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)<\|eot_id\|>',
        prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt


def load_dataset(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {path}")
    return data


def save_dataset(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {path}")


# ── 主程序 ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Backward probability via key span masking')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--model_path', '-m', type=str,
                        default='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_words', type=int, default=10,
                        help='关键片段最多包含多少个自然词（默认10）')
    parser.add_argument('--max_assistant_tokens', type=int, default=512)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    diffusion_lm = LLaDADiffusionLM(args.model_path, args.device)
    verifier = BackwardProbabilityVerifier(
        diffusion_lm,
        max_words=args.max_words,
        max_assistant_tokens=args.max_assistant_tokens,
    )

    dataset = load_dataset(args.input_file)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]

    failed = []

    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        try:
            question = extract_user_question(sample)
            rollouts = sample.get('rollouts', [])

            if not question or not rollouts:
                sample['backward_result_probability'] = []
                continue

            results = verifier.verify_rollouts(question, rollouts, verbose=args.verbose)
            sample['backward_result_probability'] = results

            # 前3个样本打印供检查
            if idx < 3 or args.verbose:
                scores = [f"{r['backward_score_probability']:.4f}" for r in results]
                span = results[0].get('key_span', '') if results else ''
                print(f"Sample {idx}: key_span='{span}' scores={scores}")

        except Exception as e:
            print(f"\n❌ Sample {idx} failed: {e}")
            traceback.print_exc()
            failed.append(idx)
            sample['backward_result_probability'] = []
            sample['backward_probability_error'] = str(e)

    save_dataset(dataset, args.output_file)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    success = sum(1 for s in dataset if s.get('backward_result_probability'))
    print(f"Processed: {success}/{len(dataset)}")
    if failed:
        print(f"Failed: {failed}")

    found = sum(
        1 for s in dataset
        for r in s.get('backward_result_probability', [])[:1]
        if r.get('found_question_word', False)
    )
    if success:
        print(f"Question word hit rate: {found}/{success} = {found/success*100:.1f}%")


if __name__ == '__main__':
    main()