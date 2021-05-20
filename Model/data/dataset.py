import json
import torch
import random 
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    def __init__(self, filename, logger):
        """
        Initialization data file path and other data-related configurations 
        Read data from data file
        Preprocess the data
        """
        pass
    def __len__(self):
        """
        Dataset length
        """
        raise NotImplementedError
    def __getitem__(self, index):
        """
        Return a set of data pairs (data[index], label[index])
        """
        raise NotImplementedError
    
    @staticmethod 
    def collate_fn(batch_data):
        """
        As parameters to torch.utils.data.DataLoader, Preprocess batch_data
        """
        pass
    def __read_data(self):
        pass
    def __preprocess_data(self):
        pass 

class MLMDataset(Dataset):
    def __init__(self, filename, tokenizer, logger, max_seq_length=200):
        self.filename = filename
        self.logger = logger
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self.__read_data()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sent = self.data[idx]
        tokens, token_labels = self.random_word(sent)
        token_labels = [0] + token_labels + [0]
        
        tokens_id = self.tokenizer.build_inputs_with_special_tokens(tokens)
        
        tokens_masks = tokens_masks = [1] * len(tokens_id)
        segment_id = self.tokenizer.create_token_type_ids_from_sequences(tokens)
        return (tokens_id, tokens_masks, segment_id, token_labels)

    @staticmethod 
    def collate_fn(batch_data): 
        batch_size = len(batch_data)
        batch_data = list(zip(*batch_data))
        lengths = [len(x) for x in batch_data[0]]
        max_seq_length = max(lengths)
        tokens_id = torch.LongTensor(batch_size, max_seq_length).fill_(0)
        tokens_masks = torch.LongTensor(batch_size, max_seq_length).fill_(0)
        segment_id = torch.LongTensor(batch_size, max_seq_length).fill_(0)
        token_labels = torch.LongTensor(batch_size, max_seq_length).fill_(0)
        for i in range(batch_size):
            tokens_id[i, :len(batch_data[0][i])] = torch.LongTensor(batch_data[0][i])
            tokens_masks[i, :len(batch_data[1][i])] = torch.LongTensor(batch_data[1][i])
            segment_id[i, :len(batch_data[2][i])] = torch.LongTensor(batch_data[2][i])
            token_labels[i, :len(batch_data[3][i])] = torch.LongTensor(batch_data[3][i])

        return (tokens_id, tokens_masks, segment_id, token_labels)
    
    def random_word(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        token_labels = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15 

                if prob < 0.8: 
                    tokens[i] = self.tokenizer._token_mask_id
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.tokenizer.token2id))
                else:
                    tokens[i] = self.tokenizer.token_to_id(tokens[i])

                token_labels.append(self.tokenizer.token_to_id(token))
            else:
                tokens[i] = self.tokenizer.token_to_id(token)
                token_labels.append(0)
        if len(tokens) > self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 2)]
            token_labels = token_labels[0:(self.max_seq_length - 2)]

        return tokens, token_labels


    def __read_data(self):
        with open(self.filename, 'r+') as f:
            data = f.readlines()
        return data
        

def MLMDataLoader(filename, batch_size, tokenizer, logger, max_seq_length=200,
                    shuffle=False, lower=True, num_workers=8,
                    collate_fn=MLMDataset.collate_fn):

    dataset = MLMDataset(filename, tokenizer, logger, max_seq_length)
    data_loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            pin_memory=True, 
                            num_workers=num_workers, 
                            collate_fn=collate_fn)
    return data_loader


class ReBagDataset(Dataset):
    def __init__(self, filename, bag_size, logger, tokenizer, 
                max_seq_length=128, lower=True, entpair_as_bag=False):
        
        self.tokenizer = tokenizer
        self.filename = filename
        self.bag_size = bag_size
        self.logger = logger
        self.lower = lower
        self.max_seq_length = max_seq_length
        self.entpair_as_bag = entpair_as_bag
        self.label2id = {"NA":0,"gene_associated_with_disease":1, "disease_associated_with_tissue":2, "disease_associated_with_disease":3, "tissue_associated_with_tissue":4}
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.data = self.__read_data()

    def __len__(self):
        return len(self.bag_scope)

    def __read_data(self):
        if self.logger is not None:
            self.logger.debug("-----------read data-----------")
        f = open(self.filename)
        data = json.load(f)
        f.close()
        if self.logger is not None:
            self.logger.debug("{} has data {}".format(self.filename, len(data)))
        return self.__preprocess_data(data)

    def __preprocess_data(self, data):
        self.weights = np.ones((len(self.label2id)), dtype=np.float32)
        self.bag_scope = []
        self.name2id = {}
        self.bag_name = []
        self.facts = {}

        for idx, item in enumerate(data):
            fact = (item['subj']['cui'], item['obj']['cui'], item['relation'])
            if item['relation'] != 'NA':
                self.facts[fact] = 1
            if self.entpair_as_bag:
                name = (item['subj']['cui'], item['obj']['cui'])
            else:
                name = fact
            if name not in self.name2id:
                self.name2id[name] = len(self.name2id)
                self.bag_scope.append([])
                self.bag_name.append(name)
            self.bag_scope[self.name2id[name]].append(idx)
            self.weights[self.label2id[item['relation']]] += 1.0
        self.weights = 1.0 / (self.weights ** 0.05)
        self.weights = torch.from_numpy(self.weights)
        return data

    def get_feature(self, item):
        label = self.label2id[item['relation']]
        sentence = item['sent']
        pos_head = item['subj']['pos']
        pos_tail = item['obj']['pos']

        if pos_head[0] > pos_tail[0]:
            pos_min, pos_max = [pos_tail, pos_head]
            rev = True
        else:
            pos_min, pos_max = [pos_head, pos_tail]
            rev = False
        sent_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
        sent_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        sent_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])

        tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
        if rev:
            pos_tail = [len(sent_0), len(sent_0) + len(ent_0)]
            pos_head = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
        else:
            pos_head = [len(sent_0), len(sent_0) + len(ent_0)]
            pos_tail = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]        
        
        # tokens_id = [self.tokenizer.token_to_id(token) for token in tokens]
        # while len(tokens_id) < self.max_seq_length:
        #     tokens_id.append(self.tokenizer._token_pad_id)
        if len(tokens) > self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
        tokens_id, tokens_mask, segment_id = self.tokenizer.create_feature(tokens)
        while len(tokens_id) < self.max_seq_length:
            tokens_id.append(self.tokenizer._token_pad_id)
            tokens_mask.append(self.tokenizer._token_pad_id)
            segment_id.append(self.tokenizer._token_pad_id)
        
        pos1 = [self.tokenizer._token_pad_id]
        pos2 = [self.tokenizer._token_pad_id]
        pos1_in_index = min(pos_head[0], self.max_seq_length)
        pos2_in_index = min(pos_tail[0], self.max_seq_length)
        
        for i in range(len(tokens)):
            pos1.append(min(i - pos1_in_index + self.max_seq_length, 2 * self.max_seq_length - 1))
            pos2.append(min(i - pos2_in_index + self.max_seq_length, 2 * self.max_seq_length - 1))
        while len(pos1) < self.max_seq_length:
            pos1.append(self.tokenizer._token_pad_id)
        while len(pos2) < self.max_seq_length:
            pos2.append(self.tokenizer._token_pad_id)

        tokens_id = tokens_id[:self.max_seq_length]
        pos1 = pos1[:self.max_seq_length]
        pos2 = pos2[:self.max_seq_length]
        tokens_mask = tokens_mask[:self.max_seq_length]
        segment_id = segment_id[:self.max_seq_length]
        tokens_id = torch.tensor(tokens_id).long().unsqueeze(0) # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0) # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0) # (1, L)
        tokens_mask = torch.tensor(tokens_mask).long().unsqueeze(0) # (1, L)
        segment_id = torch.tensor(segment_id).long().unsqueeze(0) # (1, L)
        # Mask
        pcnn_mask = []
        pos_min = min(pos1_in_index, pos2_in_index)
        pos_max = max(pos1_in_index, pos2_in_index)
        for i in range(len(tokens)):
            if i <= pos_min:
                pcnn_mask.append(1)
            elif i <= pos_max:
                pcnn_mask.append(2)
            else:
                pcnn_mask.append(3)
        # Padding
        while len(pcnn_mask) < self.max_seq_length:
            pcnn_mask.append(0)
        pcnn_mask = pcnn_mask[:self.max_seq_length]
        pcnn_mask = torch.tensor(pcnn_mask).long().unsqueeze(0) # (1, L)

        return tokens_id, tokens_mask, segment_id, pos1, pos2, pcnn_mask

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag

        label = self.label2id[self.data[bag[0]]['relation']]
        seqs = None
        for instance_idx in bag:
            item = self.data[instance_idx]
            seq = list(self.get_feature(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (n, L), n is the size of bag
        
        return [label, self.bag_name[index], len(bag)] + seqs

    @staticmethod
    def collate_fn(batch_data):
        batch_data = list(zip(*batch_data))
        label, bag_name, count = batch_data[:3]
        seqs = batch_data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (sumn, L)
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(0))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs
    
    # @staticmethod
    # def collate_bag_size_fn(batch_data):
    #     batch_data = list(zip(*batch_data))
    #     label, bag_name, count = batch_data[:3]
    #     seqs = batch_data[3:]
    #     for i in range(len(seqs)):
    #         seqs[i] = torch.stack(seqs[i], 0) # (batch, bag, L)
    #     scope = [] # (B, 2)
    #     start = 0
    #     for c in count:
    #         scope.append((start, start + c))
    #         start += c
    #     label = torch.tensor(label).long() # (B)
    #     return [label, bag_name, scope] + seqs

def ReBagDataLoader(filename, batch_size, bag_size, logger, tokenizer, max_seq_length=128, 
                shuffle=False, lower=True, num_workers=8, entpair_as_bag=False, collate_fn=ReBagDataset.collate_fn):
    # if bag_size == 0:
    collate_fn = ReBagDataset.collate_fn
    # else:
    #     collate_fn = ReBagDataset.collate_bag_size_fn
    dataset = ReBagDataset(filename, bag_size, logger, tokenizer, max_seq_length, lower, entpair_as_bag)
    
    data_loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            pin_memory=True, 
                            num_workers=num_workers, 
                            collate_fn=collate_fn)
    return data_loader