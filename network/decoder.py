import torch
import torch.nn as nn
import math

from network.attention import *
from network.embedding import FeedForwardLayer, ResidualConnection, LayerNormalization

class DecoderLayer(nn.Module):
    def __init__(self, features: int, self_attention: MultiHeadAttentionLayer,
                        cross_attention: MultiHeadAttentionLayer,
                        ff: FeedForwardLayer,
                        dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ff = ff
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_ouput, padding_mask, look_ahead_mask):
        x = self.residual[0](x, lambda x: self.self_attention(x, x, x, look_ahead_mask))
        x = self.residual[1](x, lambda x: self.cross_attention(x, encoder_ouput, encoder_ouput, padding_mask))
        x = self.residual[2](x, self.ff)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, x, encoder_ouput, padding_mask, look_ahead_mask):
        for layer in self.layers:
            x = layer(x, encoder_ouput, padding_mask, look_ahead_mask)
        return self.layer_norm(x)