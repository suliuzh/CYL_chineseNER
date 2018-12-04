# -*- coding: UTF-8 -*-

import codecs
import re
import pdb
import pandas as pd
import numpy as np
import collections
import pickle
import os

def train2pkl():
    datas = list()
    labels = list()
    linedata=list()
    linelabel=list()
    tags = set()
    tags.add('')
    max_len = 60
    input_data = open('data/train.txt','r',encoding='utf-8')
    num=0
    linedata=[]
    linelabel=[]
    for line in input_data.readlines():
        linedata=[]
        linelabel=[]
        line = line.split()
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
        datas.append(linedata)
        labels.append(linelabel)
    input_data.close()
    print ((len(datas)))
    print ((len(labels)))
    
    import collections
    #就是把降到一维
    def flatten(x):
        result = []
        for el in x:
            if isinstance(x, collections.Iterable) and not isinstance(el, str):
                result.extend(flatten(el))
            else:   #如果是字符串的话
                result.append(el)
        return result

    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words)+1)

    
    tags = [i for i in tags]
    tag_ids = range(len(tags))

    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id["unknow"]=len(word2id)+1
    id2word[len(word2id)]="unknow"
    print(id2tag)

    
    def X_padding(words):
        ids = list(word2id[words])
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        return ids
    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)   #将word转成id
    df_data['y'] = df_data['tags'].apply(y_padding)    #将tag转成word
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))
    

    
    import pickle
    import os


    with open('data/data_train.pkl', 'wb') as outp:
	    pickle.dump(word2id, outp)
	    pickle.dump(id2word, outp)
	    pickle.dump(tag2id, outp)
	    pickle.dump(id2tag, outp)
	    pickle.dump(x, outp)
	    pickle.dump(y, outp)
	    
    print ('** Finished saving the train data.')

def data2pkl(filename):
    
    datas = list()
    labels = list()
    
    with open('data/data_train.pkl','rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    
    with open('data/'+filename+'.txt','r') as inp: 
        for line in inp.readlines():
            line = line.split() 
            tags =[]
            word_id = []  
            for item in line:
                #验证集
                tag = 'O'
                tags.append(tag2id[tag])
                word = item.split('/')[0]
                if word  in word2id:
                    word_id.append(word2id[word])
                else:
                    word_id.append(word2id["unknow"]) 
            datas.append(word_id)
            labels.append(tags)
   
    x = np.asarray(datas)
    y = np.asarray(labels)
    
    with open('data/data_'+filename+'.pkl', 'wb') as outp:
	    pickle.dump(x, outp)
	    pickle.dump(y, outp)
	    
    print ('** Finished saving the data.')
    

train2pkl()         #生成训练数据
data2pkl('dev')     #生成测试集数据
data2pkl('test.content')    #生成测验集数据