# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf

class Model:
    def __init__(self,config,embedding_pretrained,dropout_keep=1):
        
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embedding_size"]
        self.embedding_dim = config["embedding_dim"] 
        self.tag_size = config["tag_size"]
        self.pretrained = config["pretrained"]
        self.dropout_keep = dropout_keep
        self.embedding_pretrained = embedding_pretrained
        
        self.input_data = tf.placeholder(tf.int32, shape=[None,None], name="input_data") 
        self.labels = tf.placeholder(tf.int32,shape=[None,None], name="labels")
        self.embedding_placeholder = tf.placeholder(tf.float32,shape=[self.embedding_size,self.embedding_dim], name="embedding_placeholder")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")   #placeholder是需要feed的数据
        


        with tf.variable_scope("bilstm_crf") as scope:
            self._build_net()
    def _build_net(self):

        word_embeddings = tf.get_variable("word_embeddings",[self.embedding_size, self.embedding_dim])
        if self.pretrained:
            embeddings_init = word_embeddings.assign(self.embedding_pretrained)  #赋值预先训练好的向量
        
        input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)
        input_embedded = tf.nn.dropout(input_embedded,self.dropout_keep)

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                                         lstm_bw_cell, 
                                                                         input_embedded,
                                                                         sequence_length=self.sequence_lengths,
                                                                         dtype=tf.float32,
                                                                         time_major=False,
                                                                         scope=None)

        bilstm_out = tf.concat([output_fw, output_bw], axis=-1)
        bilstm_out = tf.nn.dropout(bilstm_out,self.dropout_keep)

        W = tf.get_variable(name="W", shape=[2 * self.embedding_dim, self.tag_size],initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        b = tf.get_variable(name="b", shape=[self.tag_size], dtype=tf.float32,
                        initializer=tf.zeros_initializer())


        s = tf.shape(bilstm_out)
        bilstm_out = tf.reshape(bilstm_out, [-1, 2 * self.embedding_dim])
        pred = tf.matmul(bilstm_out, W) + b
        self.out = tf.reshape(pred, [-1, s[1], self.tag_size])
 

        # Linear-CRF.
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.out, self.labels, sequence_lengths=self.sequence_lengths)

        self.loss = tf.reduce_mean(-log_likelihood)

        # Compute the viterbi sequence and score (used for prediction and test time).
        # self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(out, self.transition_params,sequence_length=self.sequence_lengths)

        # Training ops.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)





       
