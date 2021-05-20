import re
import torch 
import torch.nn as nn
from config import Config
from utils.tokenizer import Tokenizer, load_vocab, _is_punctuation
from model.crf_ner import NER
from model.pcnn import MultiInstanceLearning
"""
作为模型接口
TODO:
    建立输入与输出之间的映射，在输入中找到实体所在索引，而不是输出。
    需要用到set，tokenizer中实现

"""

class Positions(object):
    def __init__(self):
        self.start = 0
        self.end = 0
    
def process_data(sentences, tokenizer, max_length):
    sentences = sentences if isinstance(sentences, list) else [sentences]
    sentences_size = len(sentences)
    tokens_list = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens_list.append(tokens)
    # tokens_list = [ for sentence in sentences]
    tokens_feature = [tokenizer.create_feature(tokens) for tokens in tokens_list]
    tokens_feature = list(zip(*tokens_feature))

    max_len = max([len(token) for token in tokens_feature[0]])
    max_len = min([max_len, max_length])

    tokens_id = torch.LongTensor(sentences_size, max_len).fill_(tokenizer._token_pad_id)
    tokens_masks = torch.LongTensor(sentences_size, max_len).fill_(0)
    segment_id = torch.LongTensor(sentences_size, max_len).fill_(tokenizer._token_pad_id)

    for i in range(sentences_size):
        tokens_id[i, :len(tokens_feature[0][i])] = torch.LongTensor(tokens_feature[0][i])
        tokens_masks[i, :len(tokens_feature[1][i])] = torch.LongTensor(tokens_feature[1][i])
        segment_id[i, :len(tokens_feature[2][i])] = torch.LongTensor(tokens_feature[2][i])

    return tokens_id, tokens_masks, segment_id

def get_entity_pos(sentences, tokens_list, labels_list, tokenizer, res, ner_model):
    sent_id = 0
    for sent, tokens, labels in zip(sentences, tokens_list, labels_list):
        mapping = tokenizer.rematch(sent, tokens)
        entitys = []
        flag = False
        # p_pos = 0
        pos = Positions()
        # text = ''
        idx = 0
        text = []
        text_type = []
        for i, label in enumerate(labels):
            if (tokens[i][0] == '[') and (tokens[i][-1] == ']') and bool(tokens[i]):
                continue
            token = sent[mapping[i][0]:mapping[i][-1]+1]
            if tokens[i][:2] == '##':
                text[-1] += token
            else:
                if (flag == True) and (label != 'B' and label != 'I'):
                    pos.end = idx
                    for j in range(pos.start, pos.end):
                        text_type[j] += (1 << ner_model['type_id'])
                    name = ' '.join(text[pos.start:pos.end])
                    name = re.sub(r'([A-Za-z0-9]) ([^a-zA-Z0-9]) ([A-Za-z0-9])', r'\1\2\3', name)
                    entitys.append({'name': name, 'pos':[pos.start, pos.end], 'type': ner_model['entity_type']})
                    flag = False
                elif (flag == False) and (label == 'B'):
                    pos.start = idx
                    flag = True
                text.append(token)
                text_type.append(0)
                idx += 1
        if 'entity' in res[sent_id]:
            res[sent_id]['entity'].extend(entitys)
            res[sent_id]['sent_type'] = [ i+j for i, j in zip(text_type, res[sent_id]['sent_type'])]
        else:
            # res[sent_id] = {'sent': " ".join(text), 'entity': entitys}
            res[sent_id] = {'sent': text, 'entity': entitys, 'sent_type': text_type}

        sent_id += 1 
            


def get_re_feature(sentence, head, tail, tokenizer, max_seq_length=128):
    pos_head = head['pos']
    pos_tail = tail['pos']
    if pos_head[0] > pos_tail[0]:
        pos_min, pos_max = [pos_tail, pos_head]
        rev = True
    else:
        pos_min, pos_max = [pos_head, pos_tail]
        rev = False
    sent_0 = tokenizer.tokenize(sentence[:pos_min[0]])
    sent_1 = tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
    sent_2 = tokenizer.tokenize(sentence[pos_max[1]:])
    ent_0 = tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
    ent_1 = tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])

    tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
    if rev:
        pos_tail = [len(sent_0), len(sent_0) + len(ent_0)]
        pos_head = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
    else:
        pos_head = [len(sent_0), len(sent_0) + len(ent_0)]
        pos_tail = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]        
    
    if len(tokens) > max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
    tokens_id, tokens_mask, segment_id = tokenizer.create_feature(tokens)
    while len(tokens_id) < max_seq_length:
        tokens_id.append(tokenizer._token_pad_id)
        tokens_mask.append(tokenizer._token_pad_id)
        segment_id.append(tokenizer._token_pad_id)
    
    pos1 = [tokenizer._token_pad_id]
    pos2 = [tokenizer._token_pad_id]
    pos1_in_index = min(pos_head[0], max_seq_length)
    pos2_in_index = min(pos_tail[0], max_seq_length)
    
    for i in range(len(tokens)):
        pos1.append(min(i - pos1_in_index + max_seq_length, 2 * max_seq_length - 1))
        pos2.append(min(i - pos2_in_index + max_seq_length, 2 * max_seq_length - 1))
    while len(pos1) < max_seq_length:
        pos1.append(tokenizer._token_pad_id)
    while len(pos2) < max_seq_length:
        pos2.append(tokenizer._token_pad_id)

    tokens_id = tokens_id[:max_seq_length]
    pos1 = pos1[:max_seq_length]
    pos2 = pos2[:max_seq_length]
    tokens_mask = tokens_mask[:max_seq_length]
    segment_id = segment_id[:max_seq_length]
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
    while len(pcnn_mask) < max_seq_length:
        pcnn_mask.append(0)
    pcnn_mask = pcnn_mask[:max_seq_length]
    pcnn_mask = torch.tensor(pcnn_mask).long().unsqueeze(0) # (1, L)

    return tokens_id, tokens_mask, segment_id, pos1, pos2, pcnn_mask


def get_re_input(sentences, entitys, tokenizer):
    bag_scope = []
    name2id = {}
    bag_name = []
    sent_id = -1
    instance_idx = 0
    data = []
    entity_id_map = []
    for sent, entity in zip(sentences, entitys):
        sent_id += 1
        entity_id_map.append({})
        if len(entity) < 2:
            continue
        for i in range(len(entity) - 1):
            subj = entity[i]
            for j in range(i+1, len(entity)):
                obj = entity[j]
                if subj['type'] in ['gene', 'tissue'] and obj['type'] in ['gene', 'tissue']:
                    continue
                if (subj['pos'][1] >= obj['pos'][0] and subj['pos'][1] <= obj['pos'][1]) \
                or (obj['pos'][1] >= subj['pos'][0] and obj['pos'][1] <= subj['pos'][1]):
                    continue
                data.append((sent, subj, obj, sent_id))
                name = ((subj['name'], subj['type']), (obj['name'], obj['type']))
                entity_id_map[sent_id][name] = (subj['id'], obj['id'])
                if name not in name2id:
                    name2id[name] = len(name2id)
                    bag_scope.append([])
                    bag_name.append(name)
                bag_scope[name2id[name]].append(instance_idx)
                instance_idx += 1
    
    features = []
    for bag_id,bag in enumerate(bag_scope) :
        seqs = None
        for idx in bag:
            item = data[idx]
            sent_id = item[-1]
            seq = list(get_re_feature(*item[:-1], tokenizer)) 
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (n, L), n is the size of bag
        
        features.append([sent_id, bag_name[bag_id], len(bag)] + seqs)
    if len(features) == 0:
        return None
    batch_data = list(zip(*features))
    sent_id, bag_name, count = batch_data[:3]
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
    return [entity_id_map, sent_id, bag_name, scope] + seqs
    

def check(bag_name, label):
    if label == 'NA':
        return False
    else:
        subj_type = label.split('_')[0]
        obj_type = label.split('_')[-1]
        if (subj_type == bag_name[0][1] and obj_type == bag_name[1][1]) or (subj_type == bag_name[1][1] and obj_type == bag_name[0][1]):
            return True
        else:
            return False

def re_predict(sentences, entitys, label2id, cfg, pretrian_checkpoint, dense_layer_model_file, re_model_file, vocab, tokenizer, device='cpu'):
    id2label = {v:k for k, v in label2id.items()}
    dense_layer_checkpoint = torch.load(dense_layer_model_file, map_location=device)
    checkpoint = torch.load(re_model_file, map_location=device)
    model = MultiInstanceLearning(n_classes=len(label2id), vocab_size=len(vocab), device=device, bert_checkpoint=pretrian_checkpoint, dense_layer_checkpoint=dense_layer_checkpoint, **cfg.config)
    model.to(device)
    model.pcnn.load_state_dict(checkpoint['state_dict'][0])
    model.fc.load_state_dict(checkpoint['state_dict'][1])
    inputs = get_re_input(sentences, entitys, tokenizer)
    if inputs is None:
        return []
    entity_id_map, sent_id, bag_name, scopes = inputs[:4]
    args = inputs[4:]
    scopes = scopes.to(device)
    for i in range(len(args)):
        args[i] = args[i].to(device)

    with torch.no_grad():
        logits,_ = model(scopes, args, bag_size=0, label=None)
    logits = logits.argmax(dim=-1).tolist()
    relations = []
    for i, name, output in zip(sent_id, bag_name, logits):
        label = id2label[output]
        if (check(name, label)):
            subj_id = entity_id_map[i][name][0]
            obj_id = entity_id_map[i][name][1]
            # print(name, entity_id_map[i][name], label)
            tmp = {}
            if label.split('_')[0] != name[0][1]:
                tmp['source'] = obj_id
                tmp['target'] = subj_id
            else:
                tmp['source'] = subj_id
                tmp['target'] = obj_id
            tmp['sent_id'] = i
            tmp['relationship'] = label
            relations.append(tmp)

    return relations

def predict(sentences, config_file, ner_model_list,
            pretrain_model_file, dense_layer_model_file,
            re_model_file, vocab_file=None, device='cpu'):
    cfg = Config()
    cfg.load_config(config_file)
    vocab_file = cfg.config['vocab'] if vocab_file is None else vocab_file
    vocab = load_vocab(vocab_file)

    tokenizer = Tokenizer(vocab)
    pretrian_checkpoint = torch.load(pretrain_model_file, map_location=device)
    ner_label2id = {'B':0, 'I':1, 'O':2, 'X':3, '[start]':4, '[end]':5}
    re_label2id = { 
        "NA":0,
        "gene_associated_with_disease":1, 
        "disease_associated_with_tissue":2,
         "disease_associated_with_disease":3, 
         "tissue_associated_with_tissue":4
    }
    res = ner_predict(sentences, ner_label2id, cfg, ner_model_list,
                        pretrian_checkpoint, vocab, tokenizer, device)
    entitys = [item['entity'] for item in res]
    for idx, sent in enumerate(sentences) :
        relations = re_predict(sent, [entitys[idx]], re_label2id, cfg, pretrian_checkpoint,
                                dense_layer_model_file, re_model_file, vocab, tokenizer, device)
        res[idx]['relation'] = relations
    return res

def ner_predict(sentences, label2id, cfg, ner_model_list,
        pretrian_checkpoint, vocab, tokenizer, device='cpu'):
    id2label = {v:k for k, v in label2id.items()}
    model = NER(n_tag=len(label2id), tag2id=label2id, vocab_size=len(vocab), device=device, checkpoint=pretrian_checkpoint, **cfg.config)
    model.to(device)
    tokens_id, tokens_masks, segment_id = process_data(sentences, tokenizer, cfg.config['max_position_embeddings'])
    tokens_id = tokens_id.to(device)
    tokens_masks = tokens_masks.to(device)
    segment_id = segment_id.to(device)

    encoder_output, _ = model.bert(tokens_id, tokens_masks, segment_id)
    encoder_output = model.dropout(encoder_output)
    tokens_id = [tokens[mask > 0].tolist() for tokens, mask in zip(tokens_id, tokens_masks)]
    tokens_list = [ [tokenizer.id_to_token(idx) for idx in token_id] for token_id in tokens_id]
    res = [{}] * len(sentences)
    with torch.no_grad():
        for ner_model in ner_model_list:
            checkpoint = torch.load(ner_model['model_name'], map_location=device)
            model.classifier.load_state_dict(checkpoint['state_dict'][0])
            model.crf.load_state_dict(checkpoint['state_dict'][1])
            logits = model.classifier(encoder_output)
            labels_id = model.crf.viterbi_decode(logits, tokens_masks)
            labels_id = [best_path[mask > 0].tolist() for best_path, mask in zip(labels_id, tokens_masks)]
            labels_list = [ [id2label[idx] for idx in label_id] for label_id in labels_id] 
            assert len(tokens_list) == len(labels_list)
            get_entity_pos(sentences, tokens_list, labels_list, tokenizer, res, ner_model)
    for i in range(len(res)):
        item = res[i]
        res[i]['relation'] = []
        for j in range(len(item['entity'])):
            res[i]['entity'][j]['id'] = j
    del model
    return res