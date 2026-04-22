import torch
import torch.nn.functional as F

from utils.generate import generate, generate_with_confidence, generate_with_seq_log_probs


@torch.no_grad()
def sample(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1., steps=256, gen_length=256, block_size=1):
    prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                            add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    # Rollout with AR order (block_length=1)
    generated_ids = generate(model=model, prompt=prompt_ids.repeat(num_generations, 1),
                             steps=steps, gen_length=gen_length, temperature=temperature, block_length=block_size)

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return {
        'generated_ids': generated_ids,
        'prompt_len': prompt_ids.shape[1],
        'rewards': reward_fn(batch, responses, num_generations, device).float(),
    }


@torch.no_grad()
def sample_with_repeat(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1., steps=256, gen_length=256, 
                       repeat_time=1, block_size=1, apply_chat_template=True):
    if apply_chat_template:
        prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                                add_generation_prompt=True, tokenize=False)
    else:
        prompts = batch['problems']
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    # Rollout with AR order (block_length=1)
    generate_ids_list = []
    print("=======block size:{}======".format(block_size))
    for _ in range(repeat_time):
        generated_ids = generate(model=model, prompt=prompt_ids.repeat(num_generations, 1),
                                steps=steps, gen_length=gen_length, temperature=temperature, block_length=block_size)
        generate_ids_list.append(generated_ids)

    all_generated_ids = torch.cat(generate_ids_list, dim=0)
    responses = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)
    return {
        'generated_ids': all_generated_ids,
        'prompt_len': prompt_ids.shape[1],
        'rewards': reward_fn(batch, responses, num_generations*repeat_time, device).float(),
    }


@torch.no_grad()
def sample_with_repeat_seq_log_probs(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1., steps=256, gen_length=256, repeat_time=1, block_size=1):
    prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                            add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    # Rollout with AR order (block_length=1)
    generate_ids_list = []
    seq_log_probs_list = []
    print("=======block size:{}======".format(block_size))
    for _ in range(repeat_time):
        generated_ids, seq_log_probs = generate_with_seq_log_probs(model=model, prompt=prompt_ids.repeat(num_generations, 1),
                                steps=steps, gen_length=gen_length, temperature=temperature, block_length=block_size)
        generate_ids_list.append(generated_ids)
        seq_log_probs_list.extend(seq_log_probs)

    all_generated_ids = torch.cat(generate_ids_list, dim=0)
    responses = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)
    return {
        'generated_ids': all_generated_ids,
        'prompt_len': prompt_ids.shape[1],
        'rewards': reward_fn(batch, responses, seq_log_probs_list, num_generations*repeat_time, device).float(),
        'reponses': responses,
    }


@torch.no_grad()
def sample_with_repeat_rank(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1., steps=256, gen_length=256, repeat_time=1, block_size=1):
    prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                            add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    generate_ids_list = []
    print("=======block size:{}======".format(block_size))
    
    # 由于prompt_ids可能有多个batch样本，我们需要对每个样本分别处理
    batch_size = prompt_ids.shape[0]
    total_generations = num_generations * repeat_time
    
    for _ in range(repeat_time):
        # 为每个batch样本生成num_generations个序列
        repeated_prompt_ids = prompt_ids.repeat(num_generations, 1)
        generated_ids, ave_conf_list = generate_with_confidence(
            model=model, 
            prompt=repeated_prompt_ids,
            steps=steps, 
            gen_length=gen_length, 
            temperature=temperature, 
            block_length=block_size
        )
        # generated_ids shape: (num_generations * batch_size, prompt_len + gen_length)
        # ave_conf_list length: num_generations * batch_size
        avg_len = (generated_ids != 126081).sum(dim=-1).float().mean()
        print(f"avg_gen_length: {avg_len:.1f}")
        # 将生成的序列和对应的置信度配对存储
        for i in range(len(ave_conf_list)):
            generate_ids_list.append((generated_ids[i], ave_conf_list[i]))
    
    # 按置信度降序排序
    generate_ids_list = sorted(generate_ids_list, key=lambda x: x[1], reverse=True)
    print("confidence list:", [x[1] for x in generate_ids_list])
    
    # 保留置信度最高的前一半
    keep_num = int(total_generations * batch_size / 2)  # 修正：考虑batch_size
    print("retain top {} generations based on confidence".format(keep_num))
    generate_ids_list = generate_ids_list[:keep_num]
    
    # 堆叠保留的生成序列
    if generate_ids_list:
        all_generated_ids = torch.stack([x[0] for x in generate_ids_list])
        print(all_generated_ids.size())
        responses = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)
        
        # 注意：这里返回的奖励可能需要对应修改
        # 因为现在保留的序列数量可能不是num_generations*repeat_time的整数倍
        return {
            'generated_ids': all_generated_ids,
            'prompt_len': prompt_ids.shape[1],
            'rewards': reward_fn(batch, responses, len(responses), device).float() if reward_fn else None,
        }
    else:
        # 如果没有保留任何序列，返回空结果
        return {
            'generated_ids': torch.tensor([]),
            'prompt_len': prompt_ids.shape[1],
            'rewards': None,
        }


@torch.no_grad()
def sample_with_weighted_confidence(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1., steps=256, gen_length=256, repeat_time=1, block_size=1):
    prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                            add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    generate_ids_list = []
    print("=======block size:{}======".format(block_size))

    batch_size = prompt_ids.shape[0]

    for _ in range(repeat_time):
        repeated_prompt_ids = prompt_ids.repeat(num_generations, 1)
        generated_ids, ave_conf_list = generate_with_confidence(
            model=model,
            prompt=repeated_prompt_ids,
            steps=steps,
            gen_length=gen_length,
            temperature=temperature,
            block_length=block_size
        )
        avg_len = (generated_ids != 126081).sum(dim=-1).float().mean()
        print(f"avg_gen_length: {avg_len:.1f}")
        for i in range(len(ave_conf_list)):
            generate_ids_list.append((generated_ids[i], ave_conf_list[i]))

    print("confidence list:", [x[1] for x in generate_ids_list])

    if generate_ids_list:
        all_generated_ids = torch.stack([x[0] for x in generate_ids_list])
        all_confidences = [x[1] for x in generate_ids_list]
        print(all_generated_ids.size())
        responses = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)

        return {
            'generated_ids': all_generated_ids,
            'prompt_len': prompt_ids.shape[1],
            'rewards': reward_fn(
                batch, responses, num_generations * repeat_time, device,
                confidences=all_confidences
            ).float() if reward_fn else None,
        }
    else:
        return {
            'generated_ids': torch.tensor([]),
            'prompt_len': prompt_ids.shape[1],
            'rewards': None,
        }

def logprob_loss(model, inputs, valid_samples, eps=0.2, gain=1.0, temperature=1., accelerator=None,
                 gen_length=256, mask_id=126336):
    advantages, generated_ids, prompt_len = inputs['advantages'], inputs['generated_ids'], inputs['prompt_len']
    # print(advantages.size(), generated_ids.size())
    batch_size, device = advantages.shape[0], generated_ids.device
    prompt_ids, completion_ids = generated_ids[:, :prompt_len], generated_ids[:, prompt_len:]
    # print(prompt_ids.size(), completion_ids.size())
    valid_samples = accelerator.gather(valid_samples).float().mean().item()
    scale = gain / gen_length / (valid_samples + 1e-5)

    for t in range(gen_length):
        # Construct input with AR masking (Past=Observed, Future=Masked)
        x = torch.cat([prompt_ids, completion_ids[:, :t],
                       torch.full((batch_size, gen_length - t), mask_id, device=device, dtype=generated_ids.dtype)], dim=1)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            logits = model(x).logits / temperature

        # Compute log probability of next token
        log_prob = F.log_softmax(logits[:, prompt_len + t, :].float(), dim=-1)
        token_log_prob = log_prob.gather(-1, completion_ids[:, t:t+1]).squeeze(-1)

        ratio = (token_log_prob - token_log_prob.detach()).exp()
        clipped_ratio = ratio.clamp(1 - eps, 1 + eps)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        accelerator.backward(loss.mul(scale).sum())

    return {
        "reward": accelerator.gather(inputs['rewards'].detach()).mean().item(),
        "valid_samples": valid_samples,
    }

def logprob_loss_reinforce(model, inputs, valid_samples, gain=1.0, accelerator=None,
                 gen_length=256, mask_id=126336, temperature=1.):
    advantages, generated_ids, prompt_len = inputs['advantages'], inputs['generated_ids'], inputs['prompt_len']
    batch_size, device = advantages.shape[0], generated_ids.device
    prompt_ids, completion_ids = generated_ids[:, :prompt_len], generated_ids[:, prompt_len:]
    
    valid_samples = accelerator.gather(valid_samples).float().mean().item()
    scale = gain / gen_length / (valid_samples + 1e-5)

    for t in range(gen_length):
        x = torch.cat([prompt_ids, completion_ids[:, :t],
                       torch.full((batch_size, gen_length - t), mask_id, device=device, dtype=generated_ids.dtype)], dim=1)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            logits = model(x).logits / temperature

        log_prob = F.log_softmax(logits[:, prompt_len + t, :].float(), dim=-1)
        token_log_prob = log_prob.gather(-1, completion_ids[:, t:t+1]).squeeze(-1)

        # REINFORCE：直接用log_prob * advantages，去掉ratio和clip
        loss = -token_log_prob * advantages

        accelerator.backward(loss.mul(scale).sum())


def compute_group_advantages(rewards, group_size):
    mean = rewards.view(group_size, -1).mean(dim=0).repeat(group_size)
    std = rewards.view(group_size, -1).std(dim=0).repeat(group_size)
    return (rewards - mean) / (std + 1e-4)


def compute_group_advantages_rloo(rewards, group_size):
    rewards_grouped = rewards.view(-1, group_size)  # [num_problems, group_size]
    
    # RLOO baseline：每个样本的baseline是其他样本的均值
    rloo_baseline = (rewards_grouped.sum(dim=-1, keepdim=True) - rewards_grouped) / (group_size - 1)
    advantages_grouped = rewards_grouped - rloo_baseline
    
    # std过滤：组内所有reward相同则跳过
    std = rewards_grouped.std(dim=-1, keepdim=True)
    advantages_grouped = torch.where(
        std.expand_as(advantages_grouped) < 1e-4,
        torch.zeros_like(advantages_grouped),
        advantages_grouped / (std.expand_as(advantages_grouped) + 1e-4)  # 除以std稳定量级
    )
    
    return advantages_grouped.view(-1)