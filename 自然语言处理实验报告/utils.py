# coding=utf-8
import re
import numpy as np
from tqdm import tqdm
import numpy as np
from metrics import * 
from numpy import  random
import tensorflow as tf
import pandas as pd
def gen_batch(data, batch_size, shuffle=False):
    if shuffle:  # 是否混洗数据
        random.shuffle(data)
        # random.shuffle(y_data)
        pass
    seqs, labels = [], []
    for (sent_, tag_) in data:
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(tag_)
    if len(seqs) != 0:  #不够batch的情况
        yield seqs, labels
    
def get_feed(seq,labels):
    max_len = max(map(lambda x: len(list(x)), seq))
    seq_res = []
    label_res =[]
    seq_length = []
    #取行
    for s in seq:
        s= list(s)
        s_ = s[:max_len] + [0] * max(max_len - len(s), 0)
        # s.extend([0]*(max_len-len(s)))
        seq_res.append(s_)
        seq_length.append(min(len(s), max_len))
    for l in labels:
        l = list(l)
        l_ = l[:max_len] + [0] * max(max_len - len(l), 0)
        label_res.append(l_)
    return seq_res,label_res,seq_length

     

            
def splist(l,s):
    return [l[i:i+s] for i in range(0,len(l),s)]    

def train(model,sess,saver,epochs,epoch_once,batch_size,data_train,data_dev,id2word,id2tag):    
    # batch_num = int(data_train.y.shape[0] / batch_size)      
    for epoch in range(epochs): 
        batches = gen_batch(data_train, batch_size, shuffle=True) 
        for seqs, labels in tqdm(batches):
            x_batch, y_batch,seq_length = get_feed(seqs, labels)  
            feed_dict = {model.input_data:x_batch, model.labels:y_batch,model.sequence_lengths:seq_length}
            pre,loss = sess.run([model.train_op,model.loss], feed_dict)
            # print('******',loss)  #这个loss就是损失率
        print('第',epoch,'次','*'*20,loss)
        path_name = "./model/model"+str(epoch+epoch_once)+".ckpt"
        print (path_name)
        saver.save(sess, path_name)
        print ("model has been saved",str(epoch+epoch_once))
        
        ##验证集
        
        dev(model,sess,data_dev,batch_size,id2tag)
        
            
            
def dev(model,sess,data_dev,batch_size,id2tag):

    label_list, seq_len_list = [], []
    batches = gen_batch(data_dev, batch_size, shuffle=False)
    for seqs, labels in tqdm(batches):
        x_batch, y_batch,seq_length = get_feed(seqs, labels)  
        feed_dict = {model.input_data:x_batch, model.labels:y_batch,model.sequence_lengths:seq_length}
        logits, transition_params = sess.run([model.out, model.transition_params], feed_dict=feed_dict)
        label_list_ = []
        for logit, seq_len in zip(logits, seq_length):
            # 调用维特比算法求最优标注序列
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
            label_list_.append(viterbi_seq)
        label_list.append(label_list_)
        seq_len_list.append(seq_length) #句子长度
    
    preds_name_list = []
    real_tags = []
    
    for labels_s in label_list:  #label_list是按照batch_size存的
        for labels in labels_s:
            preds = [id2tag[i] for i in labels]
            preds_name_list.append(preds)
    
    for (sent_, labels) in data_dev:
        preds = [id2tag[i] for i in labels]
        real_tags.append(preds)



    print(len(preds_name_list))
    print(len(real_tags))
    measure = SpanBasedF1Measure()
    measure(preds_name_list, real_tags)
    metrics = measure.get_metric()
    print('f1-measure-overall:',metrics['f1-measure-overall'])  
                        
                 
def test(output_path,model,sess,data_test,batch_size,id2tag):
    print(output_path)
    label_list, seq_len_list = [], []
    batches = gen_batch(data_test, batch_size, shuffle=False)
    for seqs, labels in tqdm(batches):
        x_batch, y_batch,seq_length = get_feed(seqs, labels)  
        feed_dict = {model.input_data:x_batch, model.labels:y_batch,model.sequence_lengths:seq_length}
        logits, transition_params = sess.run([model.out, model.transition_params], feed_dict=feed_dict)
        label_list_ = []
        for logit, seq_len in zip(logits, seq_length):
            # 调用维特比算法求最优标注序列
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
            label_list_.append(viterbi_seq)
        label_list.append(label_list_)
        seq_len_list.append(seq_length) #句子长度
    
    preds_name_list = []
    real_tags = []
    
    for labels_s in label_list:  #label_list是按照batch_size存的
        for labels in labels_s:
            preds = [id2tag[i] for i in labels]
            preds_name_list.append(preds)
    
    with open(output_path,'a',encoding='utf-8') as outp:
        for preds in preds_name_list:
            for tem in preds:
                outp.write(tem+' ')
            outp.write('\n')
    print('output successfully!')    


