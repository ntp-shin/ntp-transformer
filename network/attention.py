import torch
import torch.nn as nn
import math

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.d_k = d_model // num_heads
        
        # Create W matrix for Q, K, V and O
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]

        # (batch, num_heads, seq_len, d_k) 
        # (batch, num_heads, seq_len, seq_len)
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores +=  (1. - mask) * -1e10
        
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ v), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model)
        # (batch, seq_len, num_heads, d_k)
        # (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

        # attn_output, attn_output_weights = self.multi_head_attention(q, k, v, attn_mask=mask)
        # return attn_output
        x, self.attention_scores = MultiHeadAttentionLayer.attention(query, key, value, mask, self.dropout)
        
        # (batch, num_heads, seq_len, d_k)
        # (batch, seq_len, num_heads, d_k)
        # (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # (batch, seq_len, d_model) 
        # (batch, seq_len, d_model)
        return self.w_o(x)
