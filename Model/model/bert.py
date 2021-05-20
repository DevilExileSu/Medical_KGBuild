import torch 
import torch.nn as nn
from model.base_model import BaseModel
from model.encoder import Encoder
from model.embedding import Embeddings
from utils.util import get_activation_func


class Pooler(BaseModel):
    def __init__(self, h_dim, act='Tanh'):
        super().__init__()
        self.dense = nn.Linear(h_dim, h_dim)
        self.activation_func = get_activation_func(act)
    def forward(self, inputs):
        first_token_tensor = inputs[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation_func(pooled_output)

        return pooled_output

class Bert(BaseModel):
    def __init__(self, vocab_size, type_vocab_size, max_position_embeddings,
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler):
        
        
        super().__init__()
        self.embeddings = Embeddings(vocab_size, type_vocab_size, max_position_embeddings, 
                                    h_dim, layer_norm_eps, dropout, device)
        
        self.encoder = Encoder(n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                                dropout, device, hid_act)
        if add_pooler:
            self.pooler = Pooler(h_dim, pooler_act)
        else:
            self.pooler = None
    def forward(self, inputs, inputs_mask, token_type_ids, position_ids=None):
        embedding_output = self.embeddings(inputs, token_type_ids, position_ids)
        encoder_output = self.encoder(embedding_output, inputs_mask)
        if self.pooler is not None:
            pooled_output = self.pooler(encoder_output)
            return encoder_output, pooled_output
        else:
            return encoder_output, None



