# https://github.com/meta-llama/llama/blob/main/llama/model.py

import torch
import torch.nn as nn
import math

class Config:
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 768
        self.num_heads = 12
        self.num_layers = 12
        self.dropout_ratio = 0.1
        self.attn_probs_dropout = 0.1
        self.max_seq_len = 1024
        self.base = 10000.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def precompute_freqs_cis(args: Config):
    dim = args.d_model // args.num_heads

    # base^{-2(i-1)/d}  (i = 1, 2, ..., d/2) <=> 1.0 / (base^{2i/d})  (i = 0, 1, ..., (d/2)-1)
    freqs = 1.0 / (args.base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # create position indexes: [0, 1, ..., max_seq_len-1]
    pos_idx = torch.arange(args.max_seq_len, device=freqs.device)

    # create angle matrix using outer product: m * theta(freqs)
    freqs = torch.einsum("i,j->ij", pos_idx, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex tensor
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x): # x is projected query tensor
    # x.shape: [batch_size, seq_len, num_heads, head_dim]
    ndim = x.ndim # 4
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # i=0<=>batch_size, i=1<=>seq_len, i=2<=>num_heads, i=3<=>head_dim
    # set batch size and num_heads to 1 for broadcasting
    # keep the rest the same => shape: [1, seq_len, 1, head_dim]
    # apply the same rotation angle to all batches and all heads
    return freqs_cis.view(*shape)

def apply_rotary_emb(q, k, freqs_cis):
    # RoPE applies 2D rotation to pairs of features, treating each pair as a complex number
    # reconfigure head_dim to [head_dim//2, 2] for complex number representation
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, q_complex)

    # after applying the rotation, restore to original shape through flatten(3)
    # [batch_size, seq_len, num_heads, head_dim // 2, 2] -> [batch_size, seq_len, num_heads, head_dim]
    q_real = torch.view_as_real(q_complex * freqs_cis).flatten(3)
    k_real = torch.view_as_real(k_complex * freqs_cis).flatten(3)
    return q_real.type_as(q), k_real.type_as(k)

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.attn_weights_dropout = nn.Dropout(config.attn_probs_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("mask", torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).to(bool),
                             persistent=False)

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
        # https://spaces.ac.cn/archives/9577
        # if W_q and W_k bias=True -> RoPE + Bias = better length extrapolation

    """
        standard RoPE: Attention(m, n) = q_m^T · R_m^T · R_n · k_n
        - q_m: query vector of position m
        - k_n: key vector of position n
        - R_m: rotation matrix of position m
        - R_n: rotation matrix of position n
        - R_m^T · R_n = R_{n-m}: rotate by the difference between two positions(n, m) 
        
        if add Bias:
        RoPE + Bias: Attention(m, n) = (q_m + a)^T + · R_m^T · R_n · (k_n + b)
        - a: bias about query
        - b: bias about key
        
        Attention(m, n) = (q_m + a)^T · R_m^T · R_n · (k_n + b)
        = q_m^T · R_m^T · R_n · k_n # standard RoPE
        + a^T · R_m^T · R_n · k_n # Bias(a)-key
        + q_m^T · R_m^T · R_n · b # query-Bias(b)
        + a^T · R_m^T · R_n · b # Bias(a)-Bias(b)
        
        "a^T · R_m^T · R_n · k_n" and "q_m^T · R_m^T · R_n · b"
        -> this is the similarity between the fixed bias vector(a/b) and the transformed query/key vector.
        since the two vectors(bias vector, query/key vector) are random values, 
        the angle between them should be close to 90 degrees,
        meaning that the expected value of the inner product should be 0.
        therefore, the effect is not as strong as the first and fourth terms(standard RoPE term, Bias(a)-Bias(b) term).
        
        Bias-Bias term: a^T · R_m^T · R_n · b = a^T · R_{n-m} · b
        => adding bias introduces a term "a^T · R_{n-m} · b" to the formula, 
        which decreases with distance |n-m|, giving locality to attention.
        
        b is rotated by dθ (d=n-m), it becomes increasingly misaligned with a. 
        the value of |d| increases, the value of this term gradually decreases. 
        because the dot product value becomes smaller as the orientations of a and b become misaligned. 
        the fourth term has the property that its value decreases as the distance between two tokens increases. 
        when this decreasing term is added to the attention matrix, the attention weights between distant tokens are lowered. 
        In other words, the model is induced to focus more on nearby tokens.
        
        the Bias-Bais term forcibly reduces the scores of distant tokens, 
        the model naturally assigns higher scores to nearby tokens and lower scores to distant ones.
        this blog argues that models that learn this pattern can apply it to sequences of lengths unseen during training, thereby improving extrapolation.
        however, the reproducibility of length extrapolation results is relatively unstable, and this is believed to be closely related to model architecture and hyperparameters
    """

    def forward(self, x, freqs_cis):
        batch_size, seq_len = x.shape[:2]
        q_proj, k_proj, v_proj = self.W_q(x), self.W_k(x), self.W_v(x)

        q_heads = q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_heads = k_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_heads = v_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_heads, k_heads = apply_rotary_emb(q_heads, k_heads, freqs_cis=freqs_cis)
        q_heads, k_heads = q_heads.transpose(1, 2), k_heads.transpose(1, 2)

        q_heads = q_heads / math.sqrt(self.head_dim)
        attn_scores = q_heads @ k_heads.transpose(2, 3)

        mask_bool = self.mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask_bool, torch.finfo(attn_scores.dtype).min)

        attn_weights = self.attn_weights_dropout(self.softmax(attn_scores))

        attn_output = attn_weights @ v_heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.W_o(attn_output)
        return attn_output

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_layer = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),  # d_ff = 4 * d_model
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def forward(self, x):
        return self.ffn_layer(x)

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.attn = MHA(config)
        self.dropout_1 = nn.Dropout(config.dropout_ratio)

        self.layer_norm_2 = nn.LayerNorm(config.d_model)

        self.ffn = FFN(config)
        self.dropout_2 = nn.Dropout(config.dropout_ratio)

    def forward(self, x, freqs_cis):
        _x = x
        norm_1 = self.layer_norm_1(x)
        attn_output = self.attn(norm_1, freqs_cis=freqs_cis)
        x = _x + self.dropout_1(attn_output)

        _x = x
        norm_2 = self.layer_norm_2(x)
        ffn_output = self.ffn(norm_2)
        x = _x + self.dropout_2(ffn_output)
        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_heads = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.decoder_blocks = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.lm_heads.weight = self.tok_emb.weight

        ## pre-compute freqs
        head_dim = config.d_model // config.num_heads
        full_freqs_cis = precompute_freqs_cis(config)
        self.register_buffer("full_freqs_cis", full_freqs_cis, persistent=False)

    def forward(self, x):
        x = self.tok_emb(x)

        # rotary embeddings slicing to fit the current sequence length
        seq_len = x.size(1)
        freqs_cis = self.full_freqs_cis[:seq_len]

        for block in self.decoder_blocks:
            x = block(x, freqs_cis=freqs_cis)

        x = self.final_norm(x)
        logits = self.lm_heads(x)
        return logits