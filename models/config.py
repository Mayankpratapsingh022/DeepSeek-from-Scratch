from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    # Model architecture
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6

    # MLA configuration
    kv_lora_rank: int = 128
    q_lora_rank: int = 192
    rope_dim: int = 32

    # MoE configuration
    n_experts: int = 4
    n_experts_per_token: int = 2
    expert_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 768
    use_shared_expert: bool = True

    # MTP configuration
    mtp_num_heads: int = 2

    # Training parameters
    dropout: float = 0.1
    bias: bool = True
    aux_loss_weight: float = 0.0
    mtp_loss_weight: float = 0.3