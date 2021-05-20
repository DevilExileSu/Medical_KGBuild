import torch 
import torch.nn as nn
from collections import OrderedDict
from model.base_model import BaseModel
from model.bert import Bert

class NER(BaseModel):
    def __init__(self, n_tag, tag2id, vocab_size, 
                    type_vocab_size, max_position_embeddings,
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
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h_dim, n_tag)
        self.crf = CRF(n_tag, tag2id, device)
        
    def forward(self, inputs, inputs_mask, token_type_ids, tags=None, position_ids=None):
        encoder_output, pooled_output = self.bert(inputs, inputs_mask, token_type_ids, position_ids)
        encoder_output = self.dropout(encoder_output)
        logits = self.classifier(encoder_output)
        outputs = [logits]
        if tags is not None:
            loss = self.crf(logits, tags, inputs_mask)
            outputs =  [-1 * loss] + outputs
        return outputs
        
class CRF(BaseModel):
    def __init__(self, n_tag, tag2id, device, pad=None):
        super().__init__()
        self.n_tag = n_tag
        self.transitions = nn.Parameter(torch.randn(self.n_tag, self.n_tag))
        self.s_transitions = nn.Parameter(torch.randn(self.n_tag))
        self.e_transitions = nn.Parameter(torch.randn(self.n_tag))
        self.tag2id = tag2id
        self.device = device
        if pad is None:
            pad = n_tag
        self.pad = pad

    def forward(self, inputs, tags, mask):
        inputs = inputs.permute(1,0,2)
        tags = tags.permute(1, 0)
        mask = mask.permute(1, 0)
        score = self._compute_score(inputs, tags, mask)
        normalizer = self._compute_normalizer(inputs, mask)
        log_likelihood = score - normalizer
        return log_likelihood

    
    def _compute_score(self, inputs, tags, mask):
        # inputs: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_len, batch_size = tags.shape
        mask = mask.float()

        # 开头和结尾特殊处理, 有特殊标签转移到句子第一个标签以及，最后一个标签转移到特殊标签
        score = self.s_transitions[tags[0]] + inputs[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_len):
            score += self.transitions[tags[i-1], tags[i]] * mask[i]
            score += inputs[i, torch.arange(batch_size), tags[i]] * mask[i]

        last_tags = tags[mask.long().sum(dim=0) - 1, torch.arange(batch_size)]
        score += self.e_transitions[last_tags]

        # score = (batch_size, )
        return score

    def _compute_normalizer(self, inputs, mask):
        seq_len, batch_size, tag_size = inputs.shape
        # score (batch_size, num_tags)
        score = self.s_transitions + inputs[0]
        mask = mask.float()
        for i in range(1, seq_len):
            trans_score = self.transitions
            emissions_score = inputs[i]
            #(batch_size, num_tags, num_tags)
            next_score = score.unsqueeze(2) + emissions_score.unsqueeze(1) + trans_score
            next_score = torch.logsumexp(next_score, dim=1)
            score = mask[i].unsqueeze(1) * next_score + (1 - mask[i].unsqueeze(1)) * score

        # (batch_size, num_tags, num_tags)
        score += self.e_transitions
        #(batch_size)
        return torch.logsumexp(score, dim=1)
    
    def viterbi_decode(self, inputs, mask):
        # score 
        inputs = inputs.permute(1,0,2)
        mask = mask.permute(1, 0)
        score = self.s_transitions + inputs[0]
        seq_len, batch_size, tag_size = inputs.shape
        path = torch.zeros((seq_len, batch_size, self.n_tag), dtype=torch.long).to(self.device)
        
        for i in range(1, seq_len):
            
            trans_score = self.transitions
            emissions_score = inputs[i]
            # (batch_size, num_tags, num_tags)
            # num_tags -> num_tags 第i-1个标签到第i个标签得分
            next_score = score.unsqueeze(2) + emissions_score.unsqueeze(1) + trans_score
            #(batch_size, num_tags) 
            # i-1 -> i 对应i标签所有可能结果最优的第i-1个标签， 
            next_score, idx = next_score.max(dim=1)
            score = mask[i].unsqueeze(1) * next_score + (1 - mask[i].unsqueeze(1)) * score
            # batch_size, num_tags
            idx = mask[i].unsqueeze(1) * idx + (1 - mask[i].unsqueeze(1)) * self.pad
            path[i-1] = idx
        
        score += self.e_transitions
        _, end_idx = score.max(dim = 1)
        seq_ends = mask.long().sum(dim=0) - 1
        path = path.permute(1,0,2).contiguous()

        # 从最后一个tag开始找到最优路径
        best_path = torch.full((batch_size, seq_len), self.pad, dtype=torch.long).to(self.device)
        for i, seq_end in enumerate(seq_ends):
            best_idx = end_idx[i]
            best_path[i, seq_end] =  best_idx.item()
            for j in range(1, seq_end + 1):
                best_idx = path[i, seq_end - j, best_idx]
                best_path[i, seq_end - j] = best_idx
        return best_path