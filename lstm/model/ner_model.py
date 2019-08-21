import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood,viterbi_decode
from data_utils import batch_yield,evalaute
import numpy as np
import copy
import time

class NERModel():
    def __init__(self,args,config,word2id,label2id):
        self.optimizer = args.adam
        self.epoch_num = args.epoch
        self.embed_size=args.embed_size
        self.vocab_size=args.vocab_size
        self.hidden_dim=args.hidden_dim
        self.num_tags=args.num_tags
        self.dropout_keep_prob = args.dropout
        self.CRF=args.CRF
        self.clip_grad=args.clip
        self.config = config
        self.lr=args.learning_rate
        self.label2id=label2id
        self.word2id=word2id
        self.batch_size=args.batch_size
        self.sentence_length=args.sentence_length

    def build_model(self):
        self.add_placeholders()
        self.instantiate_weights()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def instantiate_weights(self):
        self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size])

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None,], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.learning_rate=tf.placeholder(dtype=tf.float32,shape=[],name="learning_rate")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.Embedding,dtype=tf.float32,name="_word_embeddings")
            self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,ids=self.word_ids,name="word_embeddings")

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ =tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            print("s",s)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            print("output shape",output.shape)
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags]) #shape(batch,seqslength,num_tabs)
            self.pre=tf.argmax(self.logits,axis=-1)

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            #log_likelihood 就是损失值
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
            print("here loss")


    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_op(self):
       self.init_op = tf.global_variables_initializer()

    def train(self, train,dev):
        """
        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            for epoch in range(self.epoch_num):
                print("epoch",epoch)
                start=time.time()
                self.run_one_epoch(sess, train,dev, self.label2id, epoch, saver)
                #self.run_val_epoch(sess,dev)
                end=time.time()
                print("time:",end-start)

    def run_one_epoch(self, sess, train, dev,tag2label, epoch, saver):
        """
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        TrainX, labelY = zip(*train)
        batches = batch_yield(TrainX, labelY, self.sentence_length)
        TrainX_list = []
        TrainY_list = []
        for train,label in batches:
            TrainX = list(train)
            labelY = list(label)
            TrainX_list.append(TrainX)
            TrainY_list.append(labelY)
            if len(TrainX_list)== self.batch_size: #这里是batch_size,Train_list 中每一个元素长64，即sentence_length.
                feed_dict,_ = self.get_feed_dict(TrainX_list, TrainY_list, self.lr,
                                               self.dropout_keep_prob)
                _, loss_train, step_num_,logits= sess.run([self.train_op, self.loss, self.global_step,self.pre],
                                                     feed_dict=feed_dict)
               # print("logits shape",logits.shape) #(batch,sentencee_length)
           #这是一个batchsize的loss的均值
                logits=logits.tolist()
               # print("train loss", loss_train)
                evalaute(logits,TrainY_list)


                TrainY_list.clear()
                TrainX_list.clear()
        #print("length  devc",len(dev))
        self.run_val_epoch(sess,dev)

    def run_val_epoch(self,sess,devc):
        train,label=zip(*devc)
        batches=batch_yield(train,label,self.sentence_length)
        DevcX_list = []
        DevcY_list = []
        label_real_list=[]
        print("in test")
        for train_devc,label_devc in batches:
             train_devc=list(train_devc);label_devc=list(label_devc)
             DevcX_list.append(train_devc)
             DevcY_list.append(label_devc)
             # print("length label_devc",len( DevcX_list)) #636 batchsize
             # print("length label_devc[0]",len(DevcX_list[0])) #sentence_length 64
             if len(DevcX_list)==self.batch_size: #batchsize
                     labels=copy.copy(DevcY_list)
                     feed_dict,seq_len_list=self.get_feed_dict(DevcX_list,DevcY_list,self.lr,dropout=1)
                     if self.CRF:
                         logits, transition_params = sess.run([self.logits, self.transition_params],
                                                              feed_dict=feed_dict)
                         num=0
                         batch_predict=[]
                         for logit, seq_len in zip(logits, seq_len_list): #logit的shape是batchsize，sentence_length[96,64]
                            seq_len=int(seq_len)
                            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                            #print("viterbi_seq_length",len(viterbi_seq)) #64 sentence_length,即一句话中每一个词的预测结果。
                            #print("viterbi_seq",len(viterbi_seq)) #64
                            batch_predict.append(viterbi_seq)
                         DevcX_list.clear()
                         DevcY_list.clear()
                         evalaute(batch_predict,labels)


    def get_feed_dict(self, seqs, label=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        seq_len_list=np.ones(shape=[96,])*64 #seq_len_list 维度为batchsize，记录了每一个句子的长度。
        #print("seqs.length",len(seqs)) #96
        #print("length of seq[0]",len(seqs[0])) #64
        feed_dict = {self.word_ids: seqs,
                     self.sequence_lengths: seq_len_list,
                    }
        if label is not None:
            feed_dict[self.labels]=label
        if lr is not None:
            feed_dict[self.learning_rate] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict,seq_len_list


if __name__=="__main__":
   print("Test")
