# Deep Seek - From Scratch in PyTorch

This repository contains an ongoing implementation of DeepSeek from scratch.
Currently, it focuses on building core components such as Multi-Head Latent Attention (MLA) — a memory-efficient alternative to standard Multi-Head Attention, used in models like DeepSeek-V2/V3.
MLA is designed to accelerate inference and reduce memory usage significantly during autoregressive generation.

![Multi-Head-Latent-Attention](https://github.com/user-attachments/assets/564a2bf0-ab76-4a50-ae91-2f3eadef337d)


![Mixture of Experts](https://github.com/user-attachments/assets/d7a4196d-753f-4aa5-9534-067c2a84c0ae)



![Multi Token prediction](https://github.com/user-attachments/assets/52051bc1-641e-44f4-af4e-63f64f133a64)

## Overview

Multi-Head Latent Attention (MLA) improves over standard Multi-Head Attention (MHA) by introducing a shared **latent representation** for Key and Value projections. Instead of storing separate key and value tensors for each head, MLA caches a compressed latent vector that is later projected to keys and values — reducing the Key-Value (KV) cache size while preserving per-head expressiveness.

Mixture of Experts (MoE)
MoE introduces sparsely activated expert layers that allow the model to scale its capacity without increasing compute for every token.
During training and inference, only a subset of expert networks are activated per input, enabling better specialization and efficiency.

Multi-Token Prediction
Instead of predicting one token at a time, this approach allows the model to generate multiple tokens in parallel.
It reduces the number of autoregressive steps and speeds up inference — a critical advantage in deployment settings.



### Key Features

- Implements the core logic of MLA from scratch in PyTorch.
- Includes matrix absorption trick: `W_q @ W_uk.T` to reduce redundant computation.
- Maintains modeling capacity across attention heads (unlike Multi-Query Attention).
- Supports inference with latent KV cache updates per token.
- No use of RoPE (Rotary Positional Embeddings) in this version — hence named `RopelessMLA`.

## Architecture

1. **Compression**:
   - Input embeddings are projected into a lower-dimensional latent space using `W_dkv`.

2. **KV Caching**:
   - Only the latent vector `C_KV = X @ W_dkv` is cached across time steps.
   - This reduces cache size from `n_heads × d_head × 2` to just `latent_dim`.

3. **Query Absorption**:
   - Absorbed projection `W_q @ W_uk.T` is precomputed and used to form efficient query vectors.

4. **Attention Computation**:
   - Attention scores are computed as: `Q_absorbed @ latent_cache.T`
   - Values are obtained via: `latent_cache @ W_uv`
   - Final context is: `softmax(attn_scores) @ values`, followed by `W_o`.

## Memory Efficiency

Compared to MHA:
- KV cache size is reduced by a factor of up to **4x–50x**, depending on `latent_dim` and number of heads.
- No sharing of keys/values across heads ensures that MLA preserves high modeling capacity.

## Usage

```python
from mla import RopelessMLA

model = RopelessMLA(d_model=512, n_heads=8, kv_latent_dim=256)

# x: (batch_size, seq_len, d_model)
# kv_cache: latent KV cache from previous steps
output, new_cache = model(x, kv_cache=cache, past_length=past_len)

````

## Structure

* `RopelessMLA`: Main class implementing MLA attention mechanism.
* Latent KV cache is updated dynamically with each token during inference.
* Supports context accumulation, cache visualization, and attention heatmaps.

## Why MLA?

| Method    | Cache Size          | Head Specialization | Performance |
| --------- | ------------------- | ------------------- | ----------- |
| MHA       | High (K+V per head) | ✓                   | High        |
| MQA / GQA | Low                 | ✗ (shared heads)    | Lower       |
| **MLA**   | **Low (latent)**    | ✓                   | **High**    |

# DeepSeek-V3 Implementation

A clean implementation of DeepSeek-V3 with Multi-Head Latent Attention, Mixture of Experts, and Multi-Token Prediction.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python prepare_data.py
```
This downloads TinyStories dataset and creates `train.bin` and `validation.bin` files.

### 3. Train Model
```bash
python main.py train
```
This trains the model and saves `best_deepseek_v3.pt`.

### 4. Generate Text
```bash
python run_inference.py
```
Edit prompts in `run_inference.py` to generate different text.

## Project Structure

```
DeepSeek-from-Scratch/
├── models/           # Model architecture
├── training/         # Training utilities  
├── inference/        # Text generation
├── main.py          # Main entry point
├── run_inference.py # Simple inference script
├── prepare_data.py  # Dataset preparation
└── requirements.txt # Dependencies
```

## Usage Examples

### Custom Text Generation
Edit `run_inference.py`:
```python
my_prompts = [
    "Your custom prompt here",
    "Once upon a time",
    "The future of AI is"
]
```

### Training with Different Config
Edit `training/trainer.py` to change model size, learning rate, etc.

## Model Features

- **Multi-Head Latent Attention**: 87.5% memory reduction
- **Mixture of Experts**: Auxiliary-loss-free load balancing  
- **Multi-Token Prediction**: Enhanced training

## Files Explained

- `prepare_data.py`: Downloads and tokenizes dataset
- `train.bin`, `validation.bin`: Tokenized training data
- `best_deepseek_v3.pt`: Trained model weights
- `run_inference.py`: Simple text generation script