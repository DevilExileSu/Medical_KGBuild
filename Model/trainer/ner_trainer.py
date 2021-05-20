import torch
import torch.nn as nn
from torch.nn import functional as F
from trainer.trainer import Trainer
from utils.metrics import report
from utils.util import ensure_dir, check_dir

class NerTrainer(Trainer):
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
        for idx, (inputs, masks, token_type_ids, tags, tags_masks) in enumerate(self.data_loader):
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)
            tags = tags.to(self.device)
            token_type_ids = token_type_ids.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(inputs, masks, token_type_ids, tags)
            loss = output[0].sum() / masks.float().sum()

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
            val_loss, f1_macro  = self._valid_epoch()
        self.logger.info('Train Epoch: {}, validation loss is : {:.3f}'.format(epoch, val_loss))
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(f1_macro)
        return val_loss, f1_macro     
    
    def _valid_epoch(self):
        self.model.eval()
        val_loss = 0
        preds = []
        labels = []
        with torch.no_grad():
            for idx, (inputs, masks, token_type_ids, tags, tags_masks) in enumerate(self.valid_data_loader):
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                seq_len = masks.float().long().sum(dim=1) - 1
                tags = tags.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                output = self.model(inputs, masks, token_type_ids, tags)
                loss, logits = output
                loss = loss.sum() / masks.float().sum()

                pred = self.model.crf.viterbi_decode(logits, masks).tolist()
                for i, item in enumerate(pred):
                    for j, p in enumerate(item):
                        if j == 0:
                            continue
                        elif j == seq_len[i] -1:
                            break
                        else:
                            preds.append(self.valid_data_loader.id2label[p])
                            labels.append(self.valid_data_loader.id2label[tags[i][j].item()])

                val_loss += loss.item()
        f1_micro, f1_macro = report(preds, labels, self.logger)
        return val_loss/len(self.valid_data_loader), f1_macro 
            
    def _save_checkpoint(self, epoch, save_best=False):
        ensure_dir(self.save_dir)
        state = {
            'epoch': epoch,
            'state_dict': (self.model.classifier.state_dict(), self.model.crf.state_dict()),
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
        self.model.classifier.load_state_dict(checkpoint['state_dict'][0])
        self.model.crf.load_state_dict(checkpoint['state_dict'][1])
        if checkpoint['config']['optimizer'] != self.cfg['optimizer']:
            self.logger.debug("Optimizer type given in config file is different from that of checkpoint."
                              "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.debug("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))