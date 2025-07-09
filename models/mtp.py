import torch
import torch.nn as nn
from .layers import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe import MoELayer

class MultiTokenPredictionHead(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.depth = depth
        self.n_embd = config.n_embd

        # Combine previous hidden state with future token embedding
        self.combine_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)

        # Normalization
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)

        # Transformer components
        self.attn = MultiHeadLatentAttention(config)
        self.mlp = MoELayer(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, prev_hidden, future_token_embed):
        # Normalize inputs
        prev_norm = self.norm1(prev_hidden)
        future_norm = self.norm2(future_token_embed)

        # Combine representations
        combined = torch.cat([prev_norm, future_norm], dim=-1)
        hidden = self.combine_proj(combined)

        # Process through transformer components
        hidden = hidden + self.attn(self.attn_norm(hidden))
        hidden = hidden + self.mlp(self.mlp_norm(hidden))

        return hidden