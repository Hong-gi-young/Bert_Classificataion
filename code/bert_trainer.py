import torch
import torch.nn.utils as torch_utils

from ignite.engine import Events

from utils import get_grad_norm, get_parameter_norm
from trainer import Trainer, MyEngine
import torchmetrics
import numpy as np
VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class EngineForBert(MyEngine):
    def __init__(self, func, model, crit, optimizer, scheduler, config):
        self.scheduler = scheduler
        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def train(engine, mini_batch):
        print('미니배치 갯수',len(mini_batch['input_ids']))
        engine.model.train()
        engine.optimizer.zero_grad()
        
        x, y = mini_batch['input_ids'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device)  
        mask = mini_batch['attention_mask']
        mask = mask.to(engine.device)
        
        x = x[:,:engine.config.max_length]
        
        y_hat = engine.model(x, attention_mask = mask).logits # 현재 y_hat은 hidden 값이다. 크로스엔트로피에 hs와 실제 값을 넣어준다
        loss = engine.crit(y_hat,y)    
        loss.backward()
        
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            metric_f1 = torchmetrics.F1Score(num_classes=24).to(engine.device)
            metric_recall = torchmetrics.Recall(num_classes=24).to(engine.device)
            f1 = metric_f1(y_hat,y).item()
            recall = metric_recall(y_hat,y).item()
        else:
            accuracy = 0

        
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))
        
        engine.optimizer.step()
        engine.scheduler.step()
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'F1_score' : float(f1),
            'Recall' : float(recall),
            '|param|':p_norm, #높을수록 good
            '|g_param|': g_norm, #낮을수록 good
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()
        
        with torch.no_grad():
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device) 
            mask = mini_batch['attention_mask']
            mask = mask.to(engine.device)

            x = x[:, :engine.config.max_length]

            # Take feed-forward
            y_hat = engine.model(x, attention_mask=mask).logits
            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
                metric_f1 = torchmetrics.F1Score(num_classes=24).to(engine.device)
                metric_recall = torchmetrics.Recall(num_classes=24).to(engine.device)
                f1 = metric_f1(y_hat,y).item()
                recall = metric_recall(y_hat,y).item()                
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'F1_score' : float(f1),
            'Recall' : float(recall)
        }

class BertTrainer():
    def __init__(self, config):
        self.config = config
        
    def train(
        self,
        model, crit, optimizer, scheduler,
        train_loader, valid_loader,
    ):
        train_engine = EngineForBert(
            EngineForBert.train,
            model, crit, optimizer, scheduler, self.config
        )
        validation_engine = EngineForBert(
            EngineForBert.validate,
            model, crit, optimizer, scheduler, self.config
        )
        EngineForBert.attach(
            train_engine,
            validation_engine,
            verbose = self.config.verbose
        )
        
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine, valid_loader,
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            EngineForBert.check_best,
        )
        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )
        model.load_state_dict(validation_engine.best_model)
        return model,validation_engine.best_val_acc
        
    
    