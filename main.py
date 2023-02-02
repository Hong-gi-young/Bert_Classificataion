import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import XLNetTokenizer, XLNetModel,XLNetForSequenceClassification,RobertaTokenizer, RobertaModel

import torch_optimizer as custom_optim

from bert_trainer import BertTrainer as Trainer
from dataset import TextClassificationDataset, TextClassificationCollator
from utils import *

from sklearn.model_selection import train_test_split

def define_argparser(model_name , threshfold):
    p = argparse.ArgumentParser()

    # p.add_argument('--model_fn', require =True) #저장할 모델 이름 명
    p.add_argument('--train_fn',  default='/home/hdd_data/서울/홍기영/텍스토미콘/24_데이터_ver3') 
    p.add_argument('--pretrained_model_name', type=str, default=model_name) 
    p.add_argument('--use_albert', action='store_true') 
    p.add_argument('--threshold', type=int, default=4000)
    p.add_argument('--gpu_id', type=int, default=1)
    p.add_argument('--verbose', type=int, default=2) 
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-05) 
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--valid_ratio', type=float, default=.3)
    p.add_argument('--max_length', type=int, default=128) 
    config = p.parse_args()

    return config

def get_loaders(fn, tokenizer):
    x_data,y_data, unique_labels = read_text(fn)
    x_data,y_data = preprocessing(x_data, y_data,threshold=config.threshold) # 각 성향별 
    print('기존데이터 갯수',len(x_data)) #4000개씩 * 각 성향별 24개 = 9600
    
    # x_data = stopword(x_data)
    x_data = pattern(x_data)
    # x_data = line_split(x_data)
    print('y_data',y_data,len(y_data),type(y_data))
    print('x_data',len(x_data),type(x_data))
    index_to_label={}
    for i, label in enumerate(unique_labels):
        index_to_label[i] = label  # 0:강렬한

    train_texts,valid_texts,train_labels,valid_labels = train_test_split(x_data,y_data,stratify=y_data, test_size=0.3, shuffle=True, random_state=42)
    print('train_labels',pd.DataFrame(train_labels,columns=['value']).value_counts())
    print('valid_labels',pd.DataFrame(valid_labels,columns=['value']).value_counts())
    train_loader = DataLoader(
        TextClassificationDataset(train_texts, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length), 
    )
    valid_loader = DataLoader(
        TextClassificationDataset(valid_texts, valid_labels),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )


    return train_loader, valid_loader,index_to_label

def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer

def main(config):
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    # Get dataloaders using tokenizer from untokenized corpus.
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_loader, valid_loader, index_to_label = get_loaders(config.train_fn,tokenizer)
    print("len(train_loader)",len(train_loader)) 
    print("len(valid_loader)",len(valid_loader))
    print(
        '|train| =', len(train_loader) * config.batch_size, 
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs # 5 * 3 epoch : 15
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )
    # model = RobertaModel.from_pretrained('roberta-base')
    optimizer = get_optimizer(model, config)

    crit = nn.CrossEntropyLoss()

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model,best_val_acc = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    # pretrained_model_name = config.pretrained_model_name.split('/')
    pretrained_model_name = model_name.split('/')
    pretrained_model_name = "_".join(pretrained_model_name)
    torch.save({
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, f'/home/seoul/홍기영/마케팅인텔리전스/Bert/model/24_데이터_ver3/{round(best_val_acc,6)}_{threshfold}_{pretrained_model_name}_{config.batch_size}_{config.n_epochs}_25_데이터_ver2.pth')

if __name__ == '__main__': 
    model_names = ['kykim/bert-kor-base']
    threshfolds = range(1000,6000,1000)
    threshfold = 4000
    for model_name in model_names:
        # for threshfold in threshfolds: /
        config = define_argparser(model_name , threshfold)
        main(config)