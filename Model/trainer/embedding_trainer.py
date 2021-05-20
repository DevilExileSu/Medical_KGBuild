import torch
import torch.nn as nn
from torch.nn import functional as F
from trainer.trainer import Trainer
from utils.metrics import report
from utils.util import ensure_dir, check_dir

class PreTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, cfg, logger, data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg, logger=logger)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.do_validation = self.valid_data_loader is not None
        self.device = 'cuda:0' if cfg['cuda'] else 'cpu'
        self.log_step = cfg['log_step']

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        # tokens_id, tokens_masks, segment_id, label_ids, label_masks
        for idx, (inputs, masks, token_type_ids, labels) in enumerate(self.data_loader):
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            #output: bs, max_len, vocab_size
            #label: bs, max_len
            output = self.model(inputs, masks, token_type_ids)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            labels = labels.contiguous().view(-1)

            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if idx % self.log_step:
                self.logger.info('Train Epoch: {}, {}/{} ({:.0f}%), Loss: {:.6f}'.format(epoch, 
                        idx, 
                        len(self.data_loader), 
                        idx * 100 / len(self.data_loader), 
                        loss.item()
                        ))
        self.logger.info('Train Epoch: {}, total Loss: {:.6f}, mean Loss: {:.6f}'.format(
                epoch,
                total_loss, 
                total_loss / len(self.data_loader)
                ))
        
        if self.do_validation:
            self.logger.debug("start validation")
            val_loss  = self._valid_epoch()
        self.logger.info('Train Epoch: {}, validation loss is : {:.3f}'.format(epoch, val_loss))
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)
        return val_loss, val_loss 
    
    def _valid_epoch(self):
        self.model.eval()
        val_loss = 0
        preds = []
        labels = []
        with torch.no_grad():
            for idx, (inputs, masks, token_type_ids, labels) in enumerate(self.valid_data_loader):
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                #output: bs, max_len, vocab_size
                #label: bs, max_len
                output = self.model(inputs, masks, token_type_ids)

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                labels = labels.contiguous().view(-1)

                loss = self.criterion(output, labels)


                val_loss += loss.item()
        return val_loss/len(self.valid_data_loader)
        
    def _save_checkpoint(self, epoch, save_best=False):
        ensure_dir(self.save_dir)
        state = {
            'epoch': epoch,
            'state_dict': (self.dense_layer.state_dict(), self.model.classifier.state_dict()),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg,
            'best_score': self.best_score
        }
        if save_best: 
            filename = str(self.save_dir + '/model_best.pt')
            torch.save(state, filename)
            self.logger.debug('Saving current best: {}...'.format(filename))
        else:
            filename = str(self.save_dir + '/checkpoint_epoch_{}.pt'.format(epoch))
            torch.save(state, filename)
            self.logger.debug('Saving checkpoint: {} ...'.format(filename))
        
    def _resume_checkpoint(self, path):
        check_dir(path)
        self.logger.debug('Loading checkpoint: {}...'.format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_score = checkpoint['best_score']
        
        self.model.dense_layer.load_state_dict(checkpoint['state_dict'][0])
        self.model.classifier.load_state_dict(checkpoint['state_dict'][1])

        if checkpoint['config']['optimizer'] != self.cfg['optimizer']:
            self.logger.debug("Optimizer type given in config file is different from that of checkpoint."
                              "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.debug("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))