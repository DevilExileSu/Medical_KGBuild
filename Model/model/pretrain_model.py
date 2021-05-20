import torch 
import torch.nn as nn
from model.base_model import BaseModel 
from model.bert import Bert
from collections import OrderedDict
from utils.util import get_activation_func

"""
predictions
seq_relationship
"""

class ToEmbedding(BaseModel):
    def __init__(self, embedding_dim, vocab_size, type_vocab_size, max_position_embeddings,
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler, checkpoint=None, **kwargs):
        super().__init__()

        self.bert = Bert(vocab_size, type_vocab_size, max_position_embeddings,
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler)
    
        if checkpoint is not None:
            new_state_dict = OrderedDict()
            for k1, k2 in zip(self.bert.state_dict().keys(), checkpoint.keys()):
                new_state_dict[k1] = checkpoint[k2]

            self.bert.load_state_dict(new_state_dict)
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dense_layer = nn.Linear(h_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, vocab_size)
    

    def forward(self, inputs, inputs_mask, token_type_ids, position_ids=None):
        encoder_output, pooled_output = self.bert(inputs, inputs_mask, token_type_ids, position_ids)
        # encoder_output = self.dropout(encoder_output)
        output_embedding = self.dense_layer(encoder_output)
        output_embedding = self.dropout(output_embedding)
        output = self.classifier(output_embedding)
        return output


class MLMPrediction(BaseModel):
    def __init__(self, h_dim, act, vocab_size, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(h_dim, h_dim)
        self.LayerNorm = nn.LayerNorm(h_dim, layer_norm_eps)
        self.act = get_activation_func(act)

        self.decoder = nn.Linear(h_dim, vocab_size, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.decoder_bias

    def forward(self, inputs):
        output = self.act(self.dense(inputs))
        output = self.LayerNorm(output)
        
        output = self.decoder(output) + self.decoder_bias


class NSPrediction(BaseModel):
    def __init__(self, h_dim):
        super().__init__()
        self.seq_relationship = nn.Linear(h_dim, 2)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)

class PreTraining(BaseModel):
    def __init__(self, h_dim, act, vocab_size, layer_norm_eps,):
        super().__init__()
        self.predictions = MLMPrediction(h_dim, act, vocab_size, layer_norm_eps)
        self.seq_relationship = NSPrediction(h_dim)
    def forward(self, encoder_output, pooled_output):
        mlm_output = self.predictions(encoder_output)
        nsp_output = self.seq_relationship(pooled_output)
        return mlm_output, nsp_output
        
        

class PreTrainingModel(BaseModel):
    def __init__(self, vocab_size, type_vocab_size, max_position_embeddings, 
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler=True):
        super().__init__()

        self.bert = Bert(vocab_size, type_vocab_size, max_position_embeddings, 
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler)
        self.clf = PreTraining(h_dim, hid_act, vocab_size, layer_norm_eps)

    def forward(self, inputs, inputs_mask, token_type_ids, position_ids=None):

        encoder_output, pooled_output = self.bert(inputs, inputs_mask, token_type_ids, position_ids)
        mlm_output, nsp_output = self.clf(encoder_output, pooled_output)
        return mlm_output, nsp_output