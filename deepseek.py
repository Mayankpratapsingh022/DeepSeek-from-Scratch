class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim_model: int, the dimension of the model.
        num_heads: int, the number of attention heads.
        n_local_heads: int, Number of local attention heads for distributed training.
        q_lora_rank: int, Rank for low-rank query projection.
        kv_lora_rank: int, Rank for low-rank key/value projection.
        qk_nope_head_dim:int, Dimesionality of non-positional query/key projection.
        qk_rope_head_dim:int, Dimesionality of rotatry-positional query/key projection.
        qk_head_dim:int, Dimesionality of query/key projection.
        v_head_dim:int, Dimesionality of value projection.
        softmax_scale:float, Scaling factor for softmax in attention computation.


        

    """ 
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim 
        self.qk_v_head_dim =args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm =  RMSNorm(self.q_lora_rank) 
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim) 
        self.wkv_a = Linear(self.dim,self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1*args.mscale* math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale*mscale * mscale 


        if attn_impl == "naive":
            self.register_buffer("K_cache",torch.zeros(args.max_batch_size,args.max_seq_len,self.qk_head_dim),persistent=False)
            self.register_buffer("v_cache",torch.zeros(args.max_batch_size,args.max_seq_len,self.n_local_heads,self.v_head_dim),persistent=False)
        else:
            self.register_buffer("kv_cache",torch.zeros(args.max_batch_size,args.max_seq_len,self.kv_lora_rank),persistent=False)  
            self.register_buffer("pe_cache",torch.zeros(args.max_batch_size,args.max_seq_len,self.qk_rope_head_dim),persistent=False)

    def forward(self,x:torch.Tensor,start_pos:int,freqs_cis: torch.Tensor,mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA)

        Args: 
        x (torch.Tensor): Input tensor of shape (batch_size,seq_len,dim).
        start_pos (int): Starting position for the current sequence for caching.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for RoPE. 
        mask (Optional[torch.Tensor]): Attention mask of shape (batch_size,seq_len,seq_len).

        Returns:
        torch.Tensor: Output tensor with the same shape the input.
        
        """
        bsz,seqlen, _ = x.shape()
        end_pos = start_pos + seqlen 
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else: 
            q =  self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz,seqlen,self.n_local_heads,self.qk_head_dim)
        q_nope, q_pe = torch.split(q,[self.qk_nope_head_dim,self.qk_rope_head_dim],dim=1)
        q_pe = apply_rotary_emb(q_pe,freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv,[self.kv_lora_rank,self.qk_rope_ehad_dim],dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2),freqs_cis)

        if atten_impl == "naive":
            q = torch.cat([q_nope,q_pe],dim=1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz,seqlen,self.n_local_heads,self.qk_nope_head_dim+ self.qk_v_head_dim)
            k_rope, v = torch.split(kv,[self.qk_nope_head_dim])
            k = torch.cat([k_nope,k_pe.expand(-1,-1,self.n_local_heads,-1)]),
            self.k_cache[:bsz,start_pos:end_pos] = k
            self.v_cache[:bsz,start_pos:end_post] = v
            scores = torch.einsum("bshd,bthd->hsht",q,self.k_cache[:bsz, :end_pos]) * self.softmax_scale

        else:
            wkv_b = self.wkv_b.weights if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight,self.wkv_b.scale) 
            





            







        








      
