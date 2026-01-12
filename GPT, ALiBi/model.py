import torch
import torch.nn as nn
import tiktoken 
import math

class Config:
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 768
        self.attn_heads = 12
        self.num_layers = 12
        self.dropout_ratio = 0.1
        self.attn_probs_dropout = 0.1
        self.ctx_len = 1024
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.attn_heads
        self.head_dim = self.d_model // self.num_heads
        self.attn_weights_dropout = nn.Dropout(config.attn_probs_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("mask", torch.triu(torch.ones(config.ctx_len, config.ctx_len), diagonal=1).to(bool), persistent=False) # decoder self-attention mask

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self._init_alibi_slopes() # pre-calculate slopes at model initialization time

    ## ALiBi
    def _init_alibi_slopes(self):
        # ALiBi slopes: m = 2^(-8i/n) where i is head index, n is num_heads
        # for num_heads=8: slopes = [1/2^1, 1/2^2, ..., 1/2^8]
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-8 / n)
                return [start ** i for i in range(1, n + 1)] 

            # if n is power of 2 
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n) 
            else:
                # otherwise, interpolate
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + \
                       get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

        slopes = torch.tensor(get_slopes(self.num_heads))
        # shape: [num_heads, 1, 1] for broadcasting
        self.register_buffer("alibi_slopes", slopes.view(self.num_heads, 1, 1), persistent=False) # register_buffer의 역할 -> 학습 파라미터는 아니지만 모델의 일부로 등록, 

    def _get_alibi_bias(self, seq_len):
        # create position differences matrix: [seq_len, seq_len]
        # for causal attention: bias[i,j] = -|i-j| * slope when j <= i, else -inf
        context_position = torch.arange(seq_len, device=self.alibi_slopes.device)[:, None]
        memory_position = torch.arange(seq_len, device=self.alibi_slopes.device)[None, :]
        relative_position = memory_position - context_position  # [seq_len, seq_len]

        # ALiBi adds negative bias based on distance 
        alibi_bias = self.alibi_slopes * relative_position.unsqueeze(0)
        return alibi_bias # shape: [num_heads, seq_len, seq_len]

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]

        ## linear projection
        q_proj, k_proj, v_proj = self.W_q(x), self.W_k(x), self.W_v(x)

        ## split heads and transpose
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q_heads = q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        ## calculate attention scores
        q_heads = q_heads / math.sqrt(self.head_dim)
        attn_scores = q_heads @ k_heads.transpose(2, 3) # [batch_size, num_heads, seq_len(q_len), seq_len(k_len)]

        ## apply ALiBi bias to attention scores
        alibi_bias = self._get_alibi_bias(seq_len)  # [num_heads, seq_len, seq_len]
        attn_scores = attn_scores + alibi_bias.unsqueeze(0)  # broadcast over batch dimension

        ## apply mask to attention scores
        mask_bool = self.mask[:seq_len, :seq_len]
        # masking with minimum value of the data type => ensure numerical stability
        attn_scores = attn_scores.masked_fill(mask_bool, torch.finfo(attn_scores.dtype).min)

        ## calculate attention weights
        attn_weights = self.attn_weights_dropout(self.softmax(attn_scores))

        ## calculate attention value
        attn_output = attn_weights @ v_heads # [batch_size, num_heads, seq_len(q_len), head_dim]


        ## make attention_output contiguous tensor
        attn_output = attn_output.transpose(1, 2).contiguous()

        ## concatenate heads
        #  [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, num_heads*head_dim=d_model]
        attn_output = attn_output.view(batch_size, seq_len, -1) 

        ## linear projection
        attn_output = self.W_o(attn_output)
        return attn_output


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_layer = nn.Sequential(
            nn.Linear(config.d_model, 4*config.d_model), # d_ff = 4 * d_model
            nn.GELU(),
            nn.Linear(4*config.d_model, config.d_model),
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

    def forward(self, x):
        ## Pre-LN
        _x = x
        norm_1 = self.layer_norm_1(x)
        attn_output = self.attn(norm_1)
        x = _x + self.dropout_1(attn_output) # residual connection

        _x = x
        norm_2 = self.layer_norm_2(x)
        ffn_output = self.ffn(norm_2)
        x = _x + self.dropout_2(ffn_output) # residual connection
        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_heads = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.decoder_blocks = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])

        self.lm_heads.weight = self.tok_emb.weight # weight-tying

    def forward(self, x):
        x = self.tok_emb(x)

        for block in self.decoder_blocks:
            x = block(x)

        # unlike Post-LN, Pre-LN performs normalization first, so outputs can be unstable
        # using final_norm guarantees stable logit calculation before mapping the embedding dimension to logits prediction across vocabulary
        x = self.final_norm(x)
        logits = self.lm_heads(x)
        return logits
    
def generate(model, token_ids, max_new_tokens, context_size):
    model.eval()

    for _ in range(max_new_tokens):
        input_ids = token_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:,-1,:]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True) # shape: [batch_size, 1]
        token_ids = torch.concat((token_ids, next_token_id), dim=1)

    return token_ids

def main():
    config = Config()
    model = GPTModel(config).to(config.device)

    # input text encoding
    encoding = tiktoken.get_encoding("gpt2")
    context = "The capital of South Korea is"
    token_ids = torch.tensor(encoding.encode(context), device=config.device).unsqueeze(0) # [batch_size, seq_len]

    # generate text
    generated_ids = generate(model, token_ids, max_new_tokens=100, context_size=config.ctx_len) # [batch_size, seq_len + max_new_tokens]
    print(token_ids.shape, generated_ids.shape)

    # decoding
    decoded_text = encoding.decode(generated_ids[0].tolist())
    print(f"input: {context}")
    print(f"output: {decoded_text}")

if __name__ == "__main__":
    main()