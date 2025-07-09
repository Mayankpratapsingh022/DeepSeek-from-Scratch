import torch
import tiktoken
from models import DeepSeekV3

def generate_text(model_path, config, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = DeepSeekV3(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    enc = tiktoken.get_encoding("gpt2")
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
    
    # Decode and return
    result = enc.decode(generated.squeeze().tolist())
    return result

def run_inference_examples():
    """Run inference examples with different prompts."""
    try:
        from models import DeepSeekConfig
        
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
        
        test_prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The brave knight"
        ]
        
        print("=" * 50)
        print("DEEPSEEK-V3 INFERENCE EXAMPLES")
        print("=" * 50)
        
        for prompt in test_prompts:
            result = generate_text(
                "best_deepseek_v3.pt", 
                config, 
                prompt, 
                max_tokens=80, 
                temperature=0.7, 
                top_k=40
            )
            
            print(f"\nPrompt: '{prompt}'")
            print("Generated:", result)
            print("-" * 30)
            
    except FileNotFoundError:
        print("Model file 'best_deepseek_v3.pt' not found. Please train the model first.")
    except Exception as e:
        print(f"Error during inference: {e}")