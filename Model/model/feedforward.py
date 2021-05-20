import torch 
import torch.nn as nn
from model.base_model import BaseModel   
from utils.util import get_activation_func
class PositionwiseFeedforwardLayer(BaseModel):
    def __init__(self, h_dim, pf_dim, dropout, act='GELU'):
        super().__init__()
        
        self.fc_1 = nn.Linear(h_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, h_dim)
        self.act_func = get_activation_func(act)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):

        inputs = self.act_func(self.fc_1(inputs))
        inputs = self.dropout(inputs)
        inputs = self.fc_2(inputs)
        return inputs