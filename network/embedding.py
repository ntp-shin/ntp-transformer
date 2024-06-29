import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a vector of shape (seq_len, 1)
        position = torch.arange(seq_len, 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Apply sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # -> (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # Multiplied
        self.alpha = nn.Parameter(torch.ones(1))
        # Added
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps)**2 + self.bias


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # W1 and B1
        self.linear01 = nn.Linear(d_model, d_ff)
        # W2 and B2
        self.linear02 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, seq_len, d_ff)
        return self.linear02(self.dropout(torch.relu(self.linear01(x))))