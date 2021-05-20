import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from trainer.trainer import Trainer
# from utils import scorer
from utils.util import AverageMeter
from utils.metrics import score
from utils.util import ensure_dir, check_dir
from tqdm import tqdm
import pickle


class MILTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, cfg, logger, data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg, logger=logger)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.do_validation = self.valid_data_loader is not None
        self.device = 'cuda:0' if cfg['cuda'] else 'cpu'
        self.log_step = cfg['log_step']
        self.conv_l2 = cfg['conv_l2']
        self.pooling_l2 = cfg['pooling_l2']
        self.bag_size = cfg['bag_size']


    def _train_epoch(self, epoch):
        def get_lr(optimizer):
            return [param['lr'] for param in optimizer.param_groups]
        self.model.train()
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for idx, inputs in enumerate(self.data_loader):
            for i in range(len(inputs)):
                try:
                    inputs[i] = inputs[i].to(self.device)
                except:
                    pass
            labels = inputs[0]
            bag_name = inputs[1]
            scope = inputs[2]
            args = inputs[3:]    

            outputs,_ = self.model(scope, args, bag_size=self.bag_size, label=labels)

            loss = self.criterion(outputs, labels)
            score, pred = outputs.max(-1) # (B)
            acc = float((pred == labels).long().sum()) / labels.size(0)
            pos_total = (labels != 0).long().sum()
            pos_correct = ((pred == labels).long() * (labels != 0).long()).sum()
            if pos_total > 0:
                pos_acc = float(pos_correct) / float(pos_total)
            else:
                pos_acc = 0
            
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}, {}/{} ({:.0f}%), Loss: {:.6f}, acc: {:.6f}, pos_acc: {:.6f}'.format(epoch, 
                        idx, 
                        len(self.data_loader), 
                        idx * 100 / len(self.data_loader), 
                        avg_loss.avg,
                        avg_acc.avg,
                        avg_pos_acc.avg
                        ))

        self.logger.info('Train Epoch: {}, total Loss: {:.6f}, mean Loss: {:.6f}'.format(
                epoch,
                avg_loss.sum, 
                avg_loss.avg,
                ))
        self._save_checkpoint(epoch)
        if self.do_validation:
            self.logger.debug("start validation")
            val_loss, f1 = self._valid_epoch()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)
        self.logger.info('Train Epoch: {}, current lr is : {}'.format(epoch, get_lr(self.optimizer)))
        return val_loss, f1

    def _valid_epoch(self):
        self.model.eval()
        val_loss = 0
        preds = []
        true = []
        test = []
        with torch.no_grad():
            pred_result = []
            for idx, inputs in enumerate(self.valid_data_loader):
                for i in range(len(inputs)):
                    try:
                        inputs[i] = inputs[i].to(self.device)
                    except:
                        pass
                labels = inputs[0]
                bag_name = inputs[1]
                scope = inputs[2]
                args = inputs[3:]    
                test.append(inputs)
                logits, _ = self.model(scope, args, bag_size=self.bag_size, label=None)
                val_loss += self.criterion(logits, labels)
                logits = logits.cpu().numpy()
                for i in range(len(logits)):
                    for relid in range(len(self.valid_data_loader.dataset.label2id)):
                        if self.valid_data_loader.dataset.id2label[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2], 
                                'relation': self.valid_data_loader.dataset.id2label[relid],
                                'score': logits[i][relid]
                            })
            result = score(pred_result, self.valid_data_loader.dataset.facts)

        self.logger.info('validation auc is {:.3f} validation precision is : {:.3f}, validation recall is : {:.3f}, validation f1_micro is : {:.3f}, bset scores is : {}'.format(result['auc'], result['micro_p_mean'], result['micro_r'], result['micro_f1'], self.best_score))
        return val_loss/len(self.valid_data_loader), result['micro_f1']



    def _save_checkpoint(self, epoch, save_best=False):
        ensure_dir(self.save_dir)
        state = {
            'epoch': epoch,
            'state_dict': (self.model.pcnn.state_dict(), self.model.fc.state_dict()),
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
        
        self.model.pcnn.load_state_dict(checkpoint['state_dict'][0])
        self.model.fc.load_state_dict(checkpoint['state_dict'][1])

        if checkpoint['config']['optimizer'] != self.cfg['optimizer']:
            self.logger.debug("Optimizer type given in config file is different from that of checkpoint."
                              "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.debug("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))