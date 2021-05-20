import torch
import torch.nn as nn
from model.base_model import BaseModel
from model.attention import MultiHeadAttentionLayer
from model.feedforward import PositionwiseFeedforwardLayer
from model.embedding import Embeddings


class Encoder(BaseModel):
    def __init__(self, n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, dropout, device, act):
        super().__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.device = device
        self.layer = nn.ModuleList()
        for i in range(n_layers):
            self.layer.append(EncoderLayer(h_dim, n_heads, pf_dim, layer_norm_eps, dropout, device, act))

    def forward(self, src, src_mask):
        output = src
        for i in range(self.n_layers):
            output = self.layer[i](output, src_mask)
        return output
        

class EncoderLayer(BaseModel):
    def __init__(self, h_dim, n_heads, pf_dim, layer_norm_eps, dropout, device, act):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(h_dim, n_heads, dropout, device)
        self.self_attention_dropout = nn.Dropout(dropout)
        self.self_attention_layer_norm = nn.LayerNorm(h_dim, layer_norm_eps)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(h_dim, pf_dim, dropout, act)
        self.positionwise_feedforward_dropout = nn.Dropout(dropout)
        self.positionwise_feedforward_layer_norm = nn.LayerNorm(h_dim, layer_norm_eps)

    def forward(self, src, src_mask):
        attention_output = self.self_attention(src, src, src, src_mask)
        attention_output = self.self_attention_dropout(attention_output)
        attention_output = self.self_attention_layer_norm(attention_output + src)
        
        feedforward_output = self.positionwise_feedforward(attention_output)
        feedforward_output = self.positionwise_feedforward_dropout(feedforward_output)
        output = self.positionwise_feedforward_layer_norm(feedforward_output + attention_output)
        return output