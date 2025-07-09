"""
DeepSeek-V3 Main Script

Simple entry point for training and inference.
"""

import torch
from models import DeepSeekConfig, DeepSeekV3
from training import train_model
from inference import generate_text

def demo():
    """Run a simple demo."""
    print("=" * 50)
    print("DEEPSEEK-V3 DEMO")
    print("=" * 50)
    
    # Create small model for demo
    config = DeepSeekConfig(
        vocab_size=1000,
        block_size=64,
        n_layer=3,
        n_head=4,
        n_embd=128,
        kv_lora_rank=32,
        q_lora_rank=48,
        n_experts=4,
        n_experts_per_token=2,
        mtp_num_heads=1,
        dropout=0.0
    )
    
    model = DeepSeekV3(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    test_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, total_loss, main_loss, mtp_loss = model(test_input, test_targets)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Main loss: {main_loss:.4f}")
    if mtp_loss is not None:
        print(f"MTP loss: {mtp_loss:.4f}")
        print(f"Total loss: {total_loss:.4f}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)
    
    print(f"Generated {generated.shape[1] - prompt.shape[1]} tokens")
    print("Demo completed!")

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        demo()
        return
    
    mode = sys.argv[1]
    
    if mode == "demo":
        demo()
    elif mode == "train":
        print("Starting training...")
        train_model()
    else:
        print("Usage: python main.py [demo|train|inference]")

if __name__ == "__main__":
    main()