https://www.youtube.com/watch?v=mIaWmJVrMpc 

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLA(nn.Module):
    def __init__(self, d_model, n_heads, kv_latent_dim):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads # dimension per head

        ## projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        # Compress into latent KV space
        self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False) # down-projection matrix
        # Decompress K
        self.W_uk = nn.Linear(kv_latent_dim, d_model, bias=False) # up-projection matrix
        # decompress V
        self.W_uv = nn.Linear(kv_latent_dim, d_model, bias=False) # up-projection matrix
        self.W_o = nn.Linear(d_model, d_model, bias=False) # output projection matrix

        self.ln = nn.LayerNorm(kv_latent_dim)
        self.register_buffer('absorbed_k', None) # absorbed_k = W_q @ W_uk

    def forward(self, x, kv_cache=None, past_length=0): 
        batch_size, seq_len, _ = x.shape

        ## Compute absorbed_k once: absorbed_k = W_q @ W_uk
        if self.absorbed_k is None:
            absorbed = torch.matmul(self.W_q.weight, self.W_uk.weight) 
            # absorbed.shape: (d_model, d_model) @ (d_model, kv_latent_dim) = (d_model, kv_latent_dim)
            self.absorbed_k = absorbed.view(self.n_heads, self.head_dim, -1)
            # absorbed_k.shape: (n_heads, head_dim, kv_latent_dim)

        ## Update latent KV cache 
        new_c_kv = self.ln(self.W_dkv(x)) # compress input x into latent KV space
        # new_c_kv.shape: (batch_size, seq_len, kv_latent_dim)
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # (batch_size, curr_len, kv_latent_dim)

        curr_len = c_kv.size(1) 

        ## Decompress V and split heads
        values = self.W_uv(c_kv) # (batch_size, curr_len, d_model)
        values = values.view(batch_size, curr_len, self.n_heads, self.head_dim).transpose(1, 2) # (batch_size, n_heads, curr_len, head_dim)
        

        xq = x.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)

        ## Compute attention scores
        attn_scores = torch.zeros(batch_size, self.n_heads, seq_len, curr_len, device=x.device)
        for h in range(self.n_heads):
            """
            Absorption Trick
            - c_kv = x·W_dkv
            - attention socres = Q·K^T = x·W_q(W_uk^T·W_dkv^T x^T) = x·(W_q·W_uk^T)(x·W_dkv)^T = x·(W_q·W_uk^T)·c_kv^T
            x·(W_q·W_uk^T) = Q, c_kv^T = K^T
            """
            tmp = torch.matmul(xq[:,:, h], self.absorbed_k[h]) # (batch_size, seq_len, kv_latent_dim)
            attn_scores[:, h] = torch.bmm(tmp, c_kv.transpose(1, 2)) # (batch_size, seq_len, curr_len)
        
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        # apply causal mask
        mask = torch.tril(torch.ones((seq_len, curr_len), device=x.device), diagonal=past_length)  
        attn_scores = attn_scores.masked_fill(mask.view(1, 1, seq_len, curr_len)==0, torch.finfo(attn_scores.dtype).min) 

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1) # (batch_size, n_heads, seq_len, curr_len)

        # Compute attention output
        attn_output = attn_weights @ values
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
   
        return self.W_o(attn_output) #, c_kv


