import torch
import torch.nn as nn
import math

from network.encoder import *
from network.decoder import *

class ProjectLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model)
        # (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                inputs: InputEmbedding, outputs: InputEmbedding,
                position_inputs: PositionEncoding, position_outputs: PositionEncoding,
                project_layer: ProjectLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.inputs = inputs
        self.outputs = outputs
        self.position_inputs = position_inputs
        self.position_outputs = position_outputs
        self.project_layer = project_layer

    def encode(self, inputs, padding_mask):
        inputs = self.inputs(inputs)
        inputs = self.position_inputs(inputs)
        return self.encoder(inputs, padding_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.outputs(tgt)
        tgt = self.position_outputs(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.project_layer(x)

def build_transformer(inputs_size: int, outputs_size: int,
                    inputs_seq_len: int, outputs_seq_len: int,
                    d_model: int = 512,
                    number_layers: int = 6,
                    num_heads: int = 8,
                    dropout: float = 0.1,
                    d_ff: int = 2048) -> Transformer:
    """
        inputs_size: number of srouce vocabulary
        outputs_size: number of target vocab

    """
    # 1. Input Embedding
    inputs_embedding = InputEmbedding(d_model, inputs_size)
    outputs_embedding = InputEmbedding(d_model, outputs_size)

    # 2. Position Encoder
    position_inputs = PositionEncoding(d_model, inputs_seq_len, dropout)
    position_outputs = PositionEncoding(d_model, outputs_seq_len, dropout)

    # 3.1 Encoder Layers
    encoder_layers = []
    for _ in range(number_layers):
        encoder_mha = MultiHeadAttentionLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)
        encoder_ff = FeedForwardLayer(d_model, d_ff, dropout)
        encoder_layer = EncodeLayer(d_model, encoder_mha, encoder_ff, dropout)
        encoder_layers.append(encoder_layer)

    # 3.2 Decoder_layers
    decoder_layers = []
    for _ in range(number_layers):
        self_attention = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        cross_attention = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        decoder_ff = FeedForwardLayer(d_model, d_ff, dropout)

        decoder_layer = DecoderLayer(d_model, self_attention, cross_attention, decoder_ff, dropout)
        decoder_layers.append(decoder_layer)

    # 4.1 Encoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_layers))
    # 4.2 Decoder
    decoder = Decoder(d_model, nn.ModuleList(decoder_layers))
    

    # 5. Project Layer
    project_layer = ProjectLayer(d_model, outputs_size)

    # 6. Transformer
    transformer = Transformer(encoder=encoder, decoder=decoder,
                            inputs=inputs_embedding, outputs=outputs_embedding,
                            position_inputs=position_inputs, position_outputs=position_outputs,
                            project_layer=project_layer,)

    # 7 Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

