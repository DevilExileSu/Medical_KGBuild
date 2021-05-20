import torch
import torch.nn as nn
from model.base_model import BaseModel


class Embeddings(BaseModel):
    def __init__(self, vocab_size, type_vocab_size, max_position_embeddings, h_dim, layer_norm_eps, dropout, device):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, h_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, h_dim)
        self.token_type_embedding = nn.Embedding(type_vocab_size, h_dim)

        self.LayerNorm = nn.LayerNorm(h_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.device = device

    def forward(self, inputs, token_type_ids, position_ids=None):
        seq_len = inputs.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long).expand((1,-1)).to(self.device)

        embeddings = self.word_embeddings(inputs) + self.position_embeddings(position_ids) + self.token_type_embedding(token_type_ids)

        embeddings = self.LayerNorm(embeddings)

        return self.dropout(embeddings)