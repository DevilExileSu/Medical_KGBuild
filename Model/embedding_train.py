import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config,Logger
from utils.util import get_optimizer
from utils.tokenizer import Tokenizer, load_vocab
from model.pretrain_model import ToEmbedding
from trainer.embedding_trainer import PreTrainer 
from data.dataset import MLMDataLoader
import os


parser = argparse.ArgumentParser()

# dataset parameter
parser.add_argument('--train_data', type=str, default='dataset/to_embedding/train.txt')
parser.add_argument('--valid_data', type=str, default='dataset/to_embedding/test.txt')
parser.add_argument('--vocab', type=str, default='dataset/vocab.txt')
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_seq_length', type=int, default=200)

# model parameter
parser.add_argument('--h_dim', type=int, default=768)
parser.add_argument('--n_layers', type=int, default=12)
parser.add_argument('--n_heads', type=int, default=12) 
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--pf_dim', type=int, default=3072)
parser.add_argument('--type_vocab_size', type=int, default=2)
parser.add_argument('--max_position_embeddings', type=int, default=512)
parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
parser.add_argument('--hid_act', type=str, default='gelu')
parser.add_argument('--pooler_act', type=str, default='tanh')
parser.add_argument('--add_pooler', type=bool, default=True)


# Loss function and Optimizer parameter
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--l2', type=float, default=0.0)
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
train_data_loader = MLMDataLoader(cfg.config['train_data'], cfg.config['batch_size'], tokenizer, logger, cfg.config['shuffle'], cfg.config['max_seq_length'])
valid_data_loader = MLMDataLoader(cfg.config['valid_data'], cfg.config['batch_size'], tokenizer, logger, cfg.config['shuffle'], cfg.config['max_seq_length'])

# model 
"""
n_tag, tag2id, start, end, pad, vocab_size, 
                    type_vocab_size, max_position_embeddings,
                    n_layers, h_dim, n_heads, pf_dim, layer_norm_eps, 
                    dropout, device, hid_act, pooler_act, add_pooler
"""
device = 'cuda:0' if cfg.config['cuda'] else 'cpu'
checkpoint = torch.load('saved_models/biobert-base-cased-v1.1.pth')

model = ToEmbedding(embedding_dim=100, vocab_size=len(vocab), device=device, checkpoint=checkpoint,**cfg.config)
model.to(device)
logger.info(model)

# optimizer and criterion

param = filter(lambda p: p.requires_grad, model.parameters())
# param = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(cfg.config['optimizer'], param, lr=cfg.config['lr'])
lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=cfg.config['lr_decay'], patience=cfg.config['lr_decay_patience'])
criterion = nn.CrossEntropyLoss(ignore_index=0)
# NLLLoss
#trainer
trainer = PreTrainer(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg.config, logger=logger, 
                        data_loader=train_data_loader, valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler)
trainer.train()