import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config,Logger
from utils.util import get_optimizer
from utils.tokenizer import Tokenizer,load_vocab
from model.pcnn import MultiInstanceLearning
from trainer.mil_trainer import MILTrainer
from data.dataset import ReBagDataLoader
import os


parser = argparse.ArgumentParser()

# dataset parameter
parser.add_argument('--train_data', type=str, default='dataset/re/train.txt')
parser.add_argument('--valid_data', type=str, default='dataset/re/test.txt')
parser.add_argument('--vocab', type=str, default='dataset/vocab.txt')
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--bag_size', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=160)
parser.add_argument('--max_seq_length', type=int, default=200)
parser.add_argument('--entpair_as_bag', type=bool, default=False)

# model parameter
parser.add_argument('--h_dim', type=int, default=768)
parser.add_argument('--pcnn_h_dim', type=int, default=230)
parser.add_argument('--n_layers', type=int, default=12)
parser.add_argument('--n_heads', type=int, default=12) 
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--pf_dim', type=int, default=3072)
parser.add_argument('--type_vocab_size', type=int, default=2)
parser.add_argument('--max_position_embeddings', type=int, default=512)
parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
parser.add_argument('--embedding_dim', type=int, default=100)
parser.add_argument('--pos_dim', type=int, default=5)
parser.add_argument('--pooling', choices=['max', 'avg', 'sum', 'self-att', 'cnn'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--padding_size', type=int, default=1)
parser.add_argument('--hid_act', type=str, choices=['relu', 'leaky_relu', 'gelu', 'sigmoid', 'silu', 'selu', 'tanh'], default='gelu')
parser.add_argument('--pooler_act', type=str, choices=['relu', 'leaky_relu', 'gelu', 'sigmoid', 'silu', 'selu', 'tanh'], default='tanh')
parser.add_argument('--add_pooler', type=bool, default=True)


# Loss function and Optimizer parameter
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling_l2', type=float, default=0.002, help='L2-penalty for all pooling output.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--lr_decay_patience', type=int, default=6)

# train parameter
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--save_epochs', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_path', type=str, default='./saved_models/model_best.pt')
parser.add_argument('--log_step', type=int, default=20)

# other
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--config_file', type=str, default='./config.json')
parser.add_argument('--seed', type=int, default=1234)

logger = Logger()

cfg = Config(logger=logger, args=parser.parse_args())
cfg.print_config()
cfg.save_config(cfg.config['config_file'])

torch.manual_seed(cfg.config['seed'])
torch.cuda.manual_seed(cfg.config['seed'])
torch.backends.cudnn.enabled = False
np.random.seed(cfg.config['seed'])

# vocab
vocab = load_vocab('dataset/vocab.txt')
tokenizer = Tokenizer(vocab)


# data_loader
train_data_loader =  ReBagDataLoader(cfg.config['train_data'], cfg.config['batch_size'], cfg.config['bag_size'], logger, tokenizer, cfg.config['max_seq_length'], cfg.config['shuffle'], entpair_as_bag=cfg.config['entpair_as_bag'])
valid_data_loader = ReBagDataLoader(cfg.config['valid_data'], cfg.config['batch_size'], cfg.config['bag_size'], logger, tokenizer, cfg.config['max_seq_length'], cfg.config['shuffle'], entpair_as_bag=cfg.config['entpair_as_bag'])

# model 
"""
n_tag, tag2id, start, end, pad, vocab_size, 
                    type_vocab_size, max_position_embeddings,
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler
"""

device = 'cuda:0' if cfg.config['cuda'] else 'cpu'
bert_checkpoint = torch.load('saved_models/biobert-base-cased-v1.1.pth')
dense_layer_checkpoint = torch.load('saved_models/to_embedding.pt')

n_classes = len(train_data_loader.dataset.label2id)
model = MultiInstanceLearning(n_classes=n_classes, vocab_size=len(vocab), device=device, bert_checkpoint=bert_checkpoint, dense_layer_checkpoint=dense_layer_checkpoint, **cfg.config)
model.to(device)
logger.info(model)

# optimizer and criterion

param = filter(lambda p: p.requires_grad, model.parameters())
# param = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(cfg.config['optimizer'], param, lr=cfg.config['lr'])
lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=cfg.config['lr_decay'], patience=cfg.config['lr_decay_patience'])
criterion = nn.CrossEntropyLoss(weight=train_data_loader.dataset.weights.to(device))

#trainer
trainer = MILTrainer(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg.config, logger=logger, 
                    data_loader=train_data_loader, valid_data_loader=valid_data_loader, lr_scheduler=None)
trainer.train()