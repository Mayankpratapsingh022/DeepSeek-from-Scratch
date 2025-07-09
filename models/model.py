import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe import MoELayer
from .mtp import MultiTokenPredictionHead

class DeepSeekBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadLatentAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MoELayer(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DeepSeekV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.h = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Multi-Token Prediction heads
        if config.mtp_num_heads > 0:
            self.mtp_heads = nn.ModuleList([
                MultiTokenPredictionHead(config, depth)
                for depth in range(1, config.mtp_num_heads + 1)
            ])
        else:
            self.mtp_heads = None

        # Initialize weights
        self.apply(self._init_weights)

        # Special initialization for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.h:
            x = block(x)

        # Final norm
        x = self.ln_f(x)

        # Main language modeling head
        main_logits = self.lm_head(x)
        main_loss = None

        if targets is not None:
            main_loss = F.cross_entropy(
                main_logits.view(-1, main_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        # Multi-Token Prediction
        mtp_loss = None
        if self.mtp_heads is not None and targets is not None:
            mtp_losses = []
            current_hidden = x

            for depth, mtp_head in enumerate(self.mtp_heads, 1):
                if t > depth:
                    future_indices = idx[:, depth:]
                    future_embeds = self.wte(future_indices)

                    if future_embeds.size(1) < current_hidden.size(1):
                        pad_size = current_hidden.size(1) - future_embeds.size(1)
                        padding = torch.zeros(
                            b, pad_size, self.config.n_embd,
                            device=device, dtype=future_embeds.dtype
                        )
                        future_embeds = torch.cat([future_embeds, padding], dim=1)
                    elif future_embeds.size(1) > current_hidden.size(1):
                        future_embeds = future_embeds[:, :current_hidden.size(1)]

                    current_hidden = mtp_head(current_hidden, future_embeds)
                    mtp_logits = self.lm_head(current_hidden)

                    if t > depth + 1:
                        shift_logits = mtp_logits[..., :-(depth+1), :].contiguous()
                        shift_labels = targets[..., depth+1:].contiguous()

                        if shift_labels.numel() > 0:
                            mtp_loss_single = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                ignore_index=-1
                            )
                            mtp_losses.append(mtp_loss_single)

            if mtp_losses:
                mtp_loss = torch.stack(mtp_losses).mean()

        # Combine losses
        if targets is not None:
            if mtp_loss is not None:
                total_loss = main_loss + self.config.mtp_loss_weight * mtp_loss
                return main_logits, total_loss, main_loss, mtp_loss
            else:
                return main_logits, main_loss, main_loss, None
        else:
            return main_logits[:, [-1], :], None, None, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx