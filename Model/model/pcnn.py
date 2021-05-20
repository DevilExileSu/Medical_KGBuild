import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from model.base_model import BaseModel
from utils.util import get_activation_func
from model.bert import Bert 


class MultiInstanceLearning(BaseModel):
    def __init__(self, batch_size, n_classes,
                 max_seq_length, vocab_size, type_vocab_size, 
                 max_position_embeddings, embedding_dim,
                 pos_dim, kernel_size, padding_size, pcnn_h_dim,
                 n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                dropout, device, hid_act, pooler_act, add_pooler,
                bert_checkpoint=None, dense_layer_checkpoint=None, **kwargs):
        """
                    n_classes, vocab_size, 
                    type_vocab_size, max_position_embeddings,
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler, checkpoint=None,
        """
        super().__init__()
        self.bert = Bert(vocab_size, type_vocab_size, max_position_embeddings,
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler)

        self.dense_layer = nn.Linear(h_dim, embedding_dim)
        if bert_checkpoint is not None:
            new_state_dict = OrderedDict()
            for k1, k2 in zip(self.bert.state_dict().keys(), bert_checkpoint.keys()):
                new_state_dict[k1] = bert_checkpoint[k2]

            self.bert.load_state_dict(new_state_dict)
            for param in self.bert.parameters():
                param.requires_grad = False

        if dense_layer_checkpoint is not None:
            self.dense_layer.load_state_dict(dense_layer_checkpoint['state_dict'][0])
            for param in self.dense_layer.parameters():
                param.requires_grad = False

        self.pcnn = PCNN(max_seq_length, pcnn_h_dim, vocab_size, embedding_dim,
                 pos_dim, kernel_size, padding_size,
                 dropout, hid_act, device)

        self.n_classes = n_classes
        self.batch_size = batch_size
        self.device = device
        self.fc = nn.Linear(self.pcnn.h_dim, n_classes)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scope, inputs, bag_size=0, label=None):
        tokens_id, tokens_mask, segment_id, pos1, pos2, pcnn_mask = inputs
        encoder_output, _ = self.bert(tokens_id, tokens_mask, segment_id)
        encoder_output = self.dropout(encoder_output)
        output = self.dense_layer(encoder_output)

        if bag_size > 0:
            # print("before token shape {}".format(token.shape))
            tokens_id = tokens_id.view(-1, tokens_id.size(-1))
            # print("after token shape {}".format(token.shape))
            pos1 = pos1.view(-1, pos1.size(-1))
            pos2 = pos2.view(-1, pos2.size(-1))
            pcnn_mask = pcnn_mask.view(-1, pcnn_mask.size(-1))
        else:
            # print("before token shape {}".format(token.shape))
            begin, end = scope[0][0], scope[-1][1]
            tokens_id = tokens_id[begin:end, :].view(-1, tokens_id.size(-1))
            # print("after token shape {}".format(token.shape))
            pos1 = pos1[begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[begin:end, :].view(-1, pos2.size(-1))
            pcnn_mask = pcnn_mask[begin:end, :].view(-1, pcnn_mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))
        
        rep = self.pcnn(output, pos1, pos2, pcnn_mask) # (nsum, H) 

        if label is not None:
            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long().to(self.device)
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                att_mat = self.fc.weight[query] # (nsum, H)
                att_score = (rep * att_mat).sum(-1) # (nsum)

                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) # (n)
                    bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) # (n, 1) * (n, H) -> (n, H) -> (H)
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = label.size(0)
                query = label.unsqueeze(1) # (B, 1)
                att_mat = self.fc.weight[query] # (B, 1, H)
                rep = rep.view(batch_size, bag_size, -1)
                att_score = (rep * att_mat).sum(-1) # (B, bag)
                softmax_att_score = self.softmax(att_score) # (B, bag)
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1) # (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)
            bag_rep = self.dropout(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:
            if bag_size == 0:
                bag_logits = []
                att_score = torch.matmul(rep, self.fc.weight.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, (softmax)n) 
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel)) # ((each rel)N, (logit)N)
                    logit_for_each_rel = logit_for_each_rel.diag() # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits,0) # after **softmax**
            else:
                batch_size = rep.size(0) // bag_size
                att_score = torch.matmul(rep, self.fc.weight.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
                att_score = att_score.view(batch_size, bag_size, -1) # (B, bag, N)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                softmax_att_score = self.softmax(att_score.transpose(1, 2)) # (B, N, (softmax)bag)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep) # (B, N, bag) * (B, bag, H) -> (B, N, H)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2) # (B, (each rel)N)
            
        return bag_logits, att_score


class PCNN(BaseModel):

    def __init__(self, max_seq_length, h_dim, vocab_size, embedding_dim,
                 pos_dim, kernel_size, padding_size,
                 dropout, act_func, device):
        
        super().__init__()
        self.max_seq_length = max_seq_length
        self.h_dim = h_dim
        self.vocab_size = vocab_size
        
        self.act_func = get_activation_func(act_func)
        self.kernel_size = kernel_size
        self.padding_size = padding_size

        self.input_dim = embedding_dim + pos_dim * 2

        self.pos1_embedding = nn.Embedding(self.max_seq_length * 2, pos_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(self.max_seq_length * 2, pos_dim, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(self.input_dim, self.h_dim, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_seq_length)

        self.mask_embedding = nn.Embedding(4,3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100
        self.device = device
        self.h_dim *= 3

    def forward(self, inputs, e1_pos, e2_pos, mask):
        x = torch.cat([inputs, 
                       self.pos1_embedding(e1_pos),
                       self.pos2_embedding(e2_pos)], 2).to(self.device)
        x = x.transpose(1,2)
        x = self.conv(x)

        mask = 1 - self.mask_embedding(mask).transpose(1,2)
        pool1 = self.pool(self.act_func(x + self._minus * mask[:, 0:1, :])) # (B, H, 1)
        pool2 = self.pool(self.act_func(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(self.act_func(x + self._minus * mask[:, 2:3, :]))

        output = torch.cat([pool1, pool2, pool3], 1)
        output = output.squeeze(2)
        output = self.dropout(output)

        return output