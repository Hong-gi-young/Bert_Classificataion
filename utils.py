import pandas as pd
from sklearn.datasets import load_files
import re,os
from konlpy.tag import Mecab
from tqdm import tqdm
import math
# from tokenizers import SentenPieceBPETokenizer
import nltk
import numpy as np
import random
import sys
random.seed(1)

def read_text(path):
    #데이터 로드
    contents =load_files(path, shuffle=True, random_state=2022)
    labels = contents['target_names']
    x_data = contents.data
    x_data = [x.decode('utf-8') for x in x_data]
    # print('x_data',x_data,len(x_data))
    y_data = contents.target
    print('x_data',len(x_data))
    print('y_data',y_data)
    # print('y_data',y_data,len(y_data))
    return x_data, y_data, labels

def preprocessing(x_data,y_data,threshold=1000):
    # y_data = list(y_data).copy()
    x_data_total=[]
    y_data_append = []
    for idx in tqdm(range(len(x_data))): # |x_data| : 20 
        x_data_split = random.sample(x_data[idx].split('\n'),threshold) #[:threshold] -> 몇줄을 가져올건지 정함 ex) 4000이라면 각 성향별 4천 행 생김
        # print('x_data_split',len(x_data_split),x_data_split)
        # print('y_data[x]',type(y_data[idx].astype(list)),y_data[idx])

        #각 줄을 리스트로 쌓기
        x_data_total = x_data_total + x_data_split

        #y_data
        for i in range(len(x_data_split)):
            y_data_append.append(int(y_data[idx])) #.tolist()
            # print('y_data_append',y_data_append)

        # print(len(y_data_append),y_data_append)
    print('y_data_append',len(y_data_append))
    print('x_data_total',len(x_data_total))
    return x_data_total,np.array(y_data_append)
    
def pattern(documents):
    for i,document in enumerate(documents):     
        
        #한자제거
        document = re.sub('/一-龥/',"",str(document))
        
        #특수문자 제거
        document = re.sub('[!?”“♥❤%;.~★◆●▶🤗꧁꧂😡🐱♻△🥵🙄😤○😰😭♡冠☝🦺🤠_}{◆@#$-=+,/\^♧@*\"※~&ㆍ』\\\\‘|\(\)\[\]\<\>`\'…》:↑→\\’]',"",str(document))

        #모음자음제거
        document = re.sub('[ㄱ-ㅎㅏ-ㅣ]',"",str(document)) 

        #영어 제거
        document = re.sub('[A-Za-z]',"",str(document))

        # 이모지 제거
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

        document = re.sub(emoji_pattern,"",str(document))
        documents[i] = document 
            
    return documents

#텍스트 정제 (불용어 제거)
def stopword(documents):
    df = pd.read_csv('../불용어.csv', header=None, encoding='cp949')
    df[0] = df[0].apply(lambda x: x.strip())
    stopwords = df[0].to_numpy()
    # nltk.download('punkt')

    for i, document in enumerate(tqdm(documents)):
    # for i, document in enumerate(documents):
        # print(len(nltk.tokenize.word_tokenize(document)))
        clean_words = [] 
        for word in nltk.tokenize.word_tokenize(document): #document
            if word not in stopwords: #불용어 제거
                clean_words.append(word)
        documents[i] = ' '.join(clean_words)      
    return documents

def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0
    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm