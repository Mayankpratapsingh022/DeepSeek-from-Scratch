# -*- coding: utf-8 -*-
"""
DeepSeek V3 Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepSeekConfig:
    """DeepSeek V3 Model Configuration"""
    
    # Model architecture
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 6
    n_embd: int = 768
    n_head: int = 12
    
    # Multi-Head Latent Attention (MLA) config
    kv_lora_rank: int = 512
    q_lora_rank: int = 192
    rope_dim: int = 32
    
    # Mixture of Experts (MoE) config
    n_experts: int = 6
    n_experts_per_token: int = 2
    expert_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 768
    use_shared_expert: bool = True
    
    # Multi-Token Prediction (MTP)
    mtp_num_heads: int = 3
    
    # Regularization
    dropout: float = 0.1
    bias: bool = True
    
    # Loss weights
    aux_loss_weight: float = 0.0
    mtp_loss_weight: float = 0.3
    
    # Training specific
    max_seq_len: int = 1024
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.kv_lora_rank <= self.n_embd, "kv_lora_rank should be <= n_embd"
        assert self.q_lora_rank <= self.n_embd, "q_lora_rank should be <= n_embd"
        assert self.n_experts_per_token <= self.n_experts, "n_experts_per_token should be <= n_experts"


@dataclass
class TrainingConfig:
    """Training Configuration"""
    
    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    max_iters: int = 20000
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-9
    
    # Training dynamics
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 200
    
    # Checkpointing
    save_interval: int = 2000
    checkpoint_dir: str = "checkpoints"
    
    # Device and precision
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True
    
    # Logging
    log_interval: int = 100
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference Configuration"""
    
    # Generation parameters
    max_new_tokens: int = 250
    temperature: float = 0.7
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.8
    repetition_penalty: float = 1.1
    
    # Model loading
    checkpoint_path: str = "best_model.pt"
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    # Batch inference
    batch_size: int = 1