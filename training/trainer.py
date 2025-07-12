import torch
import math
from tqdm.auto import tqdm
from contextlib import nullcontext
from models import DeepSeekConfig, DeepSeekV3
from .data_loader import get_batch, estimate_loss

def train_model():
    # Configuration
    config = DeepSeekConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        kv_lora_rank=128,
        q_lora_rank=192,
        n_experts=8,
        n_experts_per_token=2,
        mtp_num_heads=1,
        dropout=0.1
    )

    # Training parameters
    learning_rate = 3e-4
    max_iters = 80000
    warmup_steps = 8000
    min_lr = 1e-5
    eval_iters = 2000
    batch_size = 28
    gradient_accumulation_steps = 4

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize model
    torch.manual_seed(42)
    model = DeepSeekV3(config)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DeepSeek-V3 model with {total_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9
    )

    # Training loop
    model.train()
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler(device, enabled=(dtype == 'float16'))

    for epoch in tqdm(range(max_iters)):
        # Evaluation
        if epoch % eval_iters == 0 and epoch != 0:
            losses = estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx)
            print(f"Epoch {epoch}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), "best_deepseek_v3.pt")

        # Training step
        X, y = get_batch("train", config, batch_size, device_type, device)

        with ctx:
            _, total_loss, main_loss, mtp_loss = model(X, y)
            loss = total_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Learning rate scheduling
        if epoch < warmup_steps:
            lr = learning_rate * (epoch + 1) / warmup_steps
        else:
            progress = (epoch - warmup_steps) / (max_iters - warmup_steps)
            lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print("Training completed!")
    return model, config