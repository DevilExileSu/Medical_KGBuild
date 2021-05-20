import numpy as np
import torch
class BaseDataLoader(object):
    """
    Nonuse torch.utils.data
    """
    def __init__(self, filename, batch_size, shuffle, logger):
        """
        Initialization data file path, batch data size, shuffle data
        Read data from data file
        Preprocess the data
        Spilt the data according to batch_size
        """
        pass
    def __len__(self):
        """
        How many batch
        """
        raise NotImplementedError
    def __getitem__(self, index):
        """
        Return batch_size data pairs
        """
        raise NotImplementedError
    def __read_data(self,):
        pass
    def __preprocess_data(self,):
        pass

class NerDataloader(object):
    def __init__(self, filename, batch_size, tokenizer, logger,shuffle=True, max_seq_length=200):
        self.logger = logger
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.filename = filename
        self.max_seq_length = max_seq_length
        self.label2id = {'B':0, 'I':1, 'O':2, 'X':3, '[start]':4, '[end]':5}
        self.id2label = {v:k for k,v in self.label2id.items()}
        self.data = self.__read_data()
        if shuffle:
            idx = np.random.permutation(len(self.data))
            self.data = [self.data[i] for i in idx]
        self.logger.debug("{} has data {}".format(self.filename, len(self.data)))
        self.data = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        batch_data = self.data[index]
        batch_size = len(batch_data)
        batch_data = list(zip(*batch_data))
        lengths = [len(x) for x in batch_data[0]]
        max_len = max(lengths)
        
        token_pad = self.tokenizer._token_pad_id
        # tokens_id, tokens_masks, segment_id,label_ids, label_masks
        tokens_id = torch.LongTensor(batch_size, max_len).fill_(token_pad)
        tokens_masks = torch.LongTensor(batch_size, max_len).fill_(0)
        segment_id = torch.LongTensor(batch_size, max_len).fill_(token_pad)
        label_ids = torch.LongTensor(batch_size, max_len).fill_(token_pad)
        label_masks = torch.LongTensor(batch_size, max_len).fill_(0)
        for i in range(batch_size):
            tokens_id[i, :len(batch_data[0][i])] = torch.LongTensor(batch_data[0][i])
            tokens_masks[i, :len(batch_data[1][i])] = torch.LongTensor(batch_data[1][i])
            segment_id[i, :len(batch_data[2][i])] = torch.LongTensor(batch_data[2][i])
            label_ids[i, :len(batch_data[3][i])] = torch.LongTensor(batch_data[3][i])
            label_masks[i, :len(batch_data[-1][i])] = torch.LongTensor(batch_data[-1][i])

        return (tokens_id, tokens_masks, segment_id, label_ids, label_masks)

    def __read_data(self):
        data = []
        with open(self.filename, 'r', encoding='utf8') as f:
            lines = f.readlines()
            words = []
            labels = []
            for line in lines:
                if line == "" or line == "\n":
                    if words:
                        data.append({"words": words, "labels":labels})
                        words = []
                        labels = []
                else:
                    word, label = line.strip().split('\t')
                    words.append(word)
                    labels.append(label)
            if words:
                data.append({'words': words, 'labels': labels})
        return self.__preprocess_data(data)
    
    def __preprocess_data(self, data):
        new_data = []
        for item in data:
            words, labels = item.values()
            # tokens = [self.tokenizer._token_start]
            tokens = []
            label_masks = [0]
            label_ids = [self.label2id['O']]

            for word, label in zip(words, labels):
                sub_words, _ = self.tokenizer.tokenize(word)
                
                tokens.extend(sub_words)
                for i in range(len(sub_words)):
                    if i == 0:
                        label_masks.append(1)
                        label_ids.append(self.label2id[label])
                    else:
                        label_masks.append(0)
                        label_ids.append(self.label2id['X'])

            if len(tokens) > self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                label_masks = label_masks[0:(self.max_seq_length - 1)]
                label_ids = label_ids[0:(self.max_seq_length - 1)]
            # tokens = [ self.tokenizer.token_to_id(token) for token in tokens]
            tokens_id, tokens_masks, segment_id = self.tokenizer.create_feature(tokens)
            label_masks.append(0)
            label_ids.append(self.label2id['O'])
            assert len(tokens_id) == len(tokens_masks)
            assert len(tokens_id) == len(segment_id)
            assert len(tokens_id) == len(label_masks)
            assert len(tokens_id) == len(label_ids)
            new_data.append((tokens_id, tokens_masks, segment_id, label_ids, label_masks))
        return new_data