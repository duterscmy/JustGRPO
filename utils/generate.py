import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    assert prompt.shape[0] == 1 or (prompt[0] == prompt).all(), \
        "generate() requires all prompts in the batch to be identical (use prompt.repeat(n, 1)). " \
        "Variable-length prompts with mask_id padding are not supported."

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@ torch.no_grad()
def generate_with_confidence(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (batch_size, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    
    Returns:
        x: Generated sequence of shape (batch_size, prompt_length + gen_length)
        mean_confidences: List of scalar mean confidence values for each sample in batch
    '''
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    # 为每个batch样本存储被unmask时的置信度
    unmask_confidences = [[] for _ in range(batch_size)]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # batch_size, seq_len

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # batch_size, seq_len
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):  # j 是batch中的样本索引
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                
                # 记录当前batch样本中被unmask的token的置信度
                for idx in select_index:
                    token_pos = idx.item()
                    if token_pos >= prompt.shape[1]:  # 只记录生成部分的token
                        unmask_confidences[j].append(x0_p[j, token_pos].item())
            
            x[transfer_index] = x0[transfer_index]
    
    # 计算每个batch样本的平均置信度
    mean_confidences = []
    for conf_list in unmask_confidences:
        if conf_list:
            mean_confidences.append(sum(conf_list) / len(conf_list))
        else:
            mean_confidences.append(0.0)

    return x, mean_confidences



@torch.no_grad()
def generate_with_confidence(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, eos_id=126081):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (batch_size, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        eos_id: End of sequence token id. If provided, only tokens before the first eos_id are considered.
    
    Returns:
        x: Generated sequence of shape (batch_size, prompt_length + gen_length)
        seq_log_probs: List of EM-RL-Sequence reward values (log probability sum) for each sample in batch
    '''
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    # 为每个batch样本存储每个位置被unmask时的对数概率 (log probability)
    # 存储格式: [(position, log_prob), ...]
    unmask_log_probs = [[] for _ in range(batch_size)]
    
    # 记录每个token被unmask的位置
    token_positions = [[] for _ in range(batch_size)]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # batch_size, seq_len

            if remasking == 'low_confidence':
                # 使用 log_softmax 获取对数概率
                log_probs = F.log_softmax(logits, dim=-1)
                # 获取选中token的对数概率
                x0_log_p = torch.squeeze(
                    torch.gather(log_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # batch_size, seq_len
            elif remasking == 'random':
                # 随机策略下，对数概率为 -log(vocab_size)
                x0_log_p = -torch.log(torch.tensor(logits.shape[-1], dtype=torch.float, device=x0.device)) * torch.ones_like(x0, dtype=torch.float)
            else:
                raise NotImplementedError(remasking)

            # 只考虑生成区域，不计算prompt部分和超出当前block的部分
            x0_log_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -float('inf')

            x0 = torch.where(mask_index, x0, x)
            # 使用对数概率作为confidence的替代
            confidence = torch.where(mask_index, x0_log_p, -float('inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):  # j 是batch中的样本索引
                # 过滤掉 -inf 的值
                valid_conf = confidence[j]
                valid_mask = valid_conf > -float('inf')
                valid_indices = torch.where(valid_mask)[0]
                valid_conf_values = valid_conf[valid_mask]
                
                k = min(num_transfer_tokens[j, i], len(valid_indices))
                if k > 0:
                    _, select_local_idx = torch.topk(valid_conf_values, k=k)
                    select_index = valid_indices[select_local_idx]
                else:
                    select_index = torch.tensor([], dtype=torch.long, device=x0.device)
                transfer_index[j, select_index] = True
                
                # 记录当前batch样本中被unmask的token的位置和对数概率
                for idx in select_index:
                    token_pos = idx.item()
                    if token_pos >= prompt.shape[1]:  # 只记录生成部分的token
                        unmask_log_probs[j].append(x0_log_p[j, token_pos].item())
                        token_positions[j].append(token_pos)
            
            x[transfer_index] = x0[transfer_index]
    
    # 计算每个batch样本的 EM-RL-Sequence 奖励值（对数概率之和，只到EOS之前）
    seq_log_probs = []
    
    for i in range(batch_size):
        if not unmask_log_probs[i]:
            seq_log_probs.append(0.0)
            continue
        
        # 按照位置排序（因为unmask的顺序可能不是按位置递增的）
        positions_and_logprobs = list(zip(token_positions[i], unmask_log_probs[i]))
        positions_and_logprobs.sort(key=lambda x: x[0])  # 按位置排序
        
        sorted_log_probs = [lp for _, lp in positions_and_logprobs]
        
        # 计算有效token长度
        valid_length = len(sorted_log_probs)
        # 如果指定了 eos_id，找到第一个 eos token 的位置
        if eos_id is not None:
            # 获取生成的序列部分
            generated_seq = x[i, prompt.shape[1]:]
            
            # 找到第一个 eos_id 的位置
            eos_positions = (generated_seq == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos_pos = eos_positions[0].item()
                # 只取 eos 之前的 token（不包括 eos 本身）
                sorted_log_probs = sorted_log_probs[:first_eos_pos]
                valid_length = len(sorted_log_probs)  # 更新有效长度
        
        # 如果有效长度为0，返回0
        if valid_length == 0:
            seq_log_probs.append(0.0)
            continue
        
        # 计算对数概率的均值（长度归一化）
        # reward = (1/T) * sum(log P(token_i | context))
        seq_log_prob = sum(sorted_log_probs) / valid_length
        seq_log_probs.append(seq_log_prob)
        print("Sample {}: log_prob = {}, valid_length = {}".format(i, seq_log_prob, valid_length))
    return x, seq_log_probs

def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    out, seq_log_probs = generate_with_confidence(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()