from email.policy import default
import sys,os
import argparse
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import openpyxl

import pandas as pd 
import numpy as np

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, TextClassificationPipeline
from utils import pattern

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=1)
    # p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--text_path', required=True)
    p.add_argument('--save_path', default='./test')
    config = p.parse_args()
    
    return config

def read_list(path):
    texts = []
    titles = []
    file_list = os.listdir(path)
    lists = [file for file in file_list if file.endswith('.txt')]
    
    for test_name in list(lists):
        # print('test/_name:',test_name)
        path2  = path+'/'+test_name.strip()
        f = open(path2, 'r', encoding='utf-8')
        text = f.read()
        title = path2.split('/')[2].split('.')[0]
        texts.append(text) 
        titles.append(title)

    return texts,titles

def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location= 'cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']
    print('saved_data[classes]',saved_data['classes'])
    x_tests,titles = read_list(path=config.text_path)
    print(x_tests,titles)
    # print('titles',titles)
    #텍스트 정제
    x_texts = pattern(x_tests)

    with torch.no_grad():
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels = len(index_to_label)
        )
        model.load_state_dict(bert_best)

        
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device 

        model.eval()

        y_hats = []
        df = pd.DataFrame()
        for idx in range(0, len(x_texts)):
            mini_batch = tokenizer(
                x_texts[idx],
                padding = True,
                truncation=True,
                return_tensors = 'pt',
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask'] #pad 처리된 곳에 attention_mask 겹치면 안됨
            mask = mask.to(device)

            y_hat = F.softmax(model(x, attention_mask = mask).logits, dim=-1)
            y_hats += [y_hat]
            df = df.append(pd.DataFrame(y_hat.cpu().numpy()), ignore_index=True) #y_hats -> 확률값

        df['name'] = pd.DataFrame(titles)
        print(df)
        df.set_index(df['name'], inplace=True, drop=True)
        df = df.iloc[:,:-1]

        df.rename(columns= index_to_label, inplace=True)
        # df.to_excel('../예측결과/%s.xlsx'%config.model_fn.split('/')[3])
        df.to_excel(f'{config.save_path}.xlsx')
        # y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)
        # probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)

        #화면에 결과 표출
        # for i in range(len(x_texts)):
        #     sys.stdout.write('%s\t%s\n' %(
        #         " ".join([index_to_label[int(indice[i][j])] for j in range(config.top_k)]), 
        #         titles[i]
        #     ))

if __name__ == '__main__':
    config = define_argparser()
    main(config)    

