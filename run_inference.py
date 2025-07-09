"""
Simple script to run inference with DeepSeek-V3
Change the prompts below and run this file to generate text!
"""

import torch
import tiktoken
from models import DeepSeekConfig, DeepSeekV3

def generate_text(prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from your prompt."""
    
    # Model configuration (same as training)
    config = DeepSeekConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        kv_lora_rank=64,
        q_lora_rank=96,
        n_experts=4,
        n_experts_per_token=2,
        mtp_num_heads=1,
        dropout=0.1
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = DeepSeekV3(config)
    try:
        model.load_state_dict(torch.load("best_deepseek_v3.pt", map_location=device))
        print("✓ Loaded trained model")
    except FileNotFoundError:
        print("⚠️  No trained model found, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    tokenizer = tiktoken.get_encoding("gpt2")
    context = torch.tensor(tokenizer.encode_ordinary(prompt)).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
    
    # Convert back to text
    result = tokenizer.decode(generated.squeeze().tolist())
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("DEEPSEEK-V3 TEXT GENERATION")
    print("=" * 60)
    
    # ============================================
    # CHANGE THESE PROMPTS TO WHATEVER YOU WANT!
    # ============================================
    
    my_prompts = [
        "Once upon a time",
        "The little girl found a magic",
        "In the future, artificial intelligence will",
        "The secret to happiness is",
        "Yesterday I went to the store and",
    ]
    
    # ============================================
    # CHANGE THESE PARAMETERS TO EXPERIMENT!
    # ============================================
    
    max_tokens = 80        # How many words to generate
    temperature = 0.8      # 0.1=boring, 0.8=balanced, 1.2=crazy
    top_k = 50            # Vocabulary limit
    
    # Generate text for each prompt
    for i, prompt in enumerate(my_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        print("-" * 40)
        
        result = generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature, 
            top_k=top_k
        )
        
        print(f"Generated: {result}")
        print()
    
    print("=" * 60)
    print("DONE! Edit the prompts above and run again!")
    print("=" * 60)