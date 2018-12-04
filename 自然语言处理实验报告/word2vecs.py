import re
import os
import pickle
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing   #开启多线程
class NERmodel(object):
    def __init__(self,train_txt,test_txt):
        print('start')
        self.train_txt=train_txt
        self.test_txt = test_txt
        self.data_train = []
        self.data_test = []
        self.data_predict = []
        self.labels = {}  #训练数据的标签类型
        self.word2id = []
        self.tag2id = []

    def get_vec(self):
        data_num = 0
        label_num = 0 
        with open(self.train_txt,'r') as f:
            contents = f.readlines()
            for line in contents:
                temp_train_end=[]
                temp_train_label=[]
                for item in line.split(' '):    
                    temp = item.split('/')
                    temp_train_end.append(temp[0])
                    temp_train_label.append(temp[1])
                self.word2id.append(temp_train_end)
                self.tag2id.append(temp_train_label)
                with open('../resource/source.txt','a') as f:
                    f.write(' '.join(temp_train_end))
                    f.write('\n')
                with open('../resource/target.txt','a') as f:
                    f.write(' '.join(temp_train_label))
                    f.write('\n')            
      

if __name__ == "__main__":
    test = NERmodel('train.txt','test.content.txt')
    test.get_vec()
    sens=LineSentence('../resource/source.txt')
    model = Word2Vec(sens, sg=1, size=100,  window=5,  min_count=5,workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format('vec.vec',binary=False)





