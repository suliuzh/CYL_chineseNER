# -*- coding: utf-8 -*
import pickle
import pdb
import re
import sys
import math

import numpy as np

import tensorflow as tf
from bilstm_crf import Model
from utils import *  #加载所有配置
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing   #开启多线程
# from data_renmin_word import data2pkl


# data2pkl()

def get_data(x,y):
    data=[]
    for i in range(len(x)):
        data.append((x[i],y[i]))
    return data



with open('data/data_train.pkl','rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)


#验证集
print('load dev data-----------------------')

with open('data/data_dev.pkl','rb') as inp:
    x_dev = pickle.load(inp)
    y_dev = pickle.load(inp)
with open('data/data_test.content.pkl','rb') as inp:
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)



data_train = get_data(x_train,y_train)
data_dev = get_data(x_dev,y_dev)
data_test = get_data(x_test,y_test)
# print(y_dev)

print('*'*20)
print ("train len:",len(x_train))

print ("word2id len", len(word2id))
print ('Creating the data generator ...')


# print(x_train)
# print(y_train)
#参数设置
epochs = 10
batch_size = 32

config = {}
config["lr"] = 0.001    #学习率
config["embedding_dim"] = 100         #词向量维度
config["sen_len"] = len(x_train[0])     #文本行数
config["batch_size"] = batch_size       #每次输入多少行
config["embedding_size"] = len(word2id)+1
config["tag_size"] = len(tag2id)
config["pretrained"]=False   #使用预先训练好的id

'''
参数记录:
31 32 0.001 100 dropout_keep=0.6
31 32 0.01 100 0.6
31 10 0.01 100

'''
with open('config.txt','a') as f:
    f.write(str(epochs)+' ')
    f.write(str(batch_size)+' ')
    for k,v in config.items():
        f.write(str(v)+' ')
    f.write('\n')


#训练词向量

# print("begin trained embedding")
# sens=LineSentence('data/source.txt')  #文件载入是词向量
# model = Word2Vec(sens, sg=1, size=50,  window=5,  min_count=5,workers=multiprocessing.cpu_count())
# model.wv.save_word2vec_format('data/vec.vec',binary=False)    
    
embedding_pre = []
if len(sys.argv)==2 and sys.argv[1]=="pretrained":
    print ("use pretrained embedding")
    config["pretrained"]=True
    word2vec = {}
    with open('data/vec.vec','r',encoding = 'utf-8') as input_data:   
        for line in input_data.readlines():
            word2vec[line.split()[0]] = line.split()[1:]

    unknow_pre = []
    unknow_pre.extend([1]*config["embedding_dim"])   #未知词向量为1
    embedding_pre.append(unknow_pre) #wordvec id 0   

    for word ,id in word2id.items():
        if word in word2vec.keys():
            embedding_pre.append(word2vec[word])  #是不是对应了id，是的
        else:
            embedding_pre.append(unknow_pre)
    
    embedding_pre = np.asarray(embedding_pre,dtype=np.float64)
    print(embedding_pre.shape)

    print ("begin to train...")
    model = Model(config,embedding_pre,dropout_keep=0.5)
    # tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        epoch_once = 0
        #加载已经训练好的模型
        ckpt = tf.train.get_checkpoint_state('./model')
        train(model,sess,saver,epochs,epoch_once,batch_size,data_train,data_dev,id2word,id2tag) 

elif len(sys.argv)==2 and sys.argv[1]=="dev":  
    print ("begin to dev...")
    model = Model(config,embedding_pre,dropout_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print ('Model not found, please train your model first')
        else:    
            path = ckpt.model_checkpoint_path
            print ('loading pre-trained model from %s.....' % path)
            saver.restore(sess, path)
            
            dev(model,sess,data_dev,batch_size,id2tag)

else: 
    print ("begin to extraction...")
    model = Model(config,embedding_pre,dropout_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print ('Model not found, please train your model first')
        else:    
            path = ckpt.model_checkpoint_path
            print ('loading pre-trained model from %s.....' % path)
            saver.restore(sess, path)
            test(sys.argv[1],model,sess,data_test,batch_size,id2tag)

    
        
     

