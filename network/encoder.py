import torch
import torch.nn as nn
import math

from network.embedding import *
from network.attention import *

class EncodeLayer(nn.Module):
    def __init__(self, mha: MultiHeadAttentionLayer, ff: FeedForwardLayer,
                dropout: float) -> None:
        super().__init__()
        self.mha = mha
        self.ff = ff
        self.residual = nn.ModuleList(ResidualConnection(dropout) for _ in range (2))

    def forward(self, x, src_mask):
        x = self.residual[0](x, lambda x: self.mha(x,x,x, src_mask))
        x = self.residual[1](x, self.ff)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)