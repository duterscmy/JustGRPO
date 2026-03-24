def test_llada_tokenization():
    """
    测试LLaDA tokenizer如何处理不同数字的tokenization
    """
    from transformers import AutoTokenizer
    
    print("="*80)
    print("Testing LLaDA Tokenization for Numbers")
    print("="*80)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', 
        trust_remote_code=True
    )
    
    # 测试用例
    test_numbers = [
        "300",
        "1.64", 
        "-5",
        "-1.5",
        "-1.45",
        "123.456",
        "-123.456",
        "0.5",
        "-0.5"
    ]
    
    print("\n数字 tokenization 分析:")
    print("-" * 60)
    
    for num in test_numbers:
        # tokenize
        tokens = tokenizer.encode(num, add_special_tokens=False)
        token_texts = [tokenizer.decode([t]) for t in tokens]
        
        print(f"\n数字: '{num}'")
        print(f"  Token数量: {len(tokens)}")
        print(f"  Token IDs: {tokens}")
        print(f"  Token文本: {token_texts}")
        
        # 显示每个token
        for i, (token_id, token_text) in enumerate(zip(tokens, token_texts)):
            print(f"    [{i}] '{token_text}' (id={token_id})")
    
    print("\n" + "="*80)
    print("测试包含数字的完整句子:")
    print("="*80)
    
    sentences = [
        "She has 300 dollars",
        "1 pound = 1.64 USD", 
        "Temperature is -5 degrees",
        "Value is -1.45",
        "The price is 123.456 dollars"
    ]
    
    for sentence in sentences:
        print(f"\n句子: '{sentence}'")
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_texts = [tokenizer.decode([t]) for t in tokens]
        
        print(f"  总token数: {len(tokens)}")
        print(f"  Token序列: {token_texts}")
        
        # 找出数字部分
        import re
        numbers = re.findall(r'-?\d+\.?\d*', sentence)
        print(f"  数字: {numbers}")
        
        for num in numbers:
            # 找到这个数字在token序列中的位置
            num_tokens = tokenizer.encode(num, add_special_tokens=False)
            print(f"    '{num}' -> {len(num_tokens)} tokens: {[tokenizer.decode([t]) for t in num_tokens]}")


if __name__ == "__main__":
    test_llada_tokenization()