import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SwiGLU

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.n_embd = config.n_embd

        # Router
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)

        # Expert MLPs
        self.experts = nn.ModuleList([
            SwiGLU(
                config.n_embd,
                config.expert_intermediate_size,
                config.n_embd,
                config.bias
            ) for _ in range(config.n_experts)
        ])

        # Shared expert
        if config.use_shared_expert:
            self.shared_expert = SwiGLU(
                config.n_embd,
                config.shared_expert_intermediate_size,
                config.n_embd,
                config.bias
            )
        else:
            self.shared_expert = None

        # Auxiliary-loss-free load balancing
        self.register_buffer('expert_bias', torch.zeros(config.n_experts))
        self.bias_update_rate = 0.001

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        # Routing phase
        router_logits = self.router(x_flat) + self.expert_bias

        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = torch.zeros_like(router_logits)
        routing_weights.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))

        # Expert computation
        output = torch.zeros_like(x_flat)
        expert_usage = torch.zeros(self.n_experts, device=x.device)

        # Process through selected experts
        for expert_idx in range(self.n_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            expert_usage[expert_idx] = expert_mask.sum().float()

            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)

                # Weight by routing probability
                weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                output[expert_mask] += expert_output * weights

        # Add shared expert output
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x_flat)
            output += shared_output

        # Auxiliary-loss-free load balancing
        if self.training:
            with torch.no_grad():
                avg_usage = expert_usage.mean()
                for i in range(self.n_experts):
                    if expert_usage[i] > avg_usage:
                        self.expert_bias[i] -= self.bias_update_rate
                    else:
                        self.expert_bias[i] += self.bias_update_rate

        return output.view(batch_size, seq_len, hidden_size)