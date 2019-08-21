import numpy as np
from tflearn.data_utils import pad_sequences
from utils.general_util import get_logger

import config
import pickle

src_file = config.FLAGS.src_file
log_path = config.FLAGS.log_path
voc_file=config.FLAGS.voc_file

#{'O': 0, 'I-ORG': 1, 'I-LOC': 2, 'I-PER': 3, 'I-MISC': 4, 'B-LOC': 5, 'B-PER': 6, 'B-MISC': 7, 'B-ORG': 8}

def voc(srcfile):
    dic_word_id={}
    dic_tag_id={}
    set_train=set()
    set_tag=set()
    with open(src_file,"r") as fr:
        for line in fr.readlines():
            raw=line.strip().split()
            train=raw[0]
            label=raw[1]
            set_train.add(train)
            set_tag.add(label)
    print("length set_train",len(set_train)) #length set_train 20186
    print("length set_tag",len(set_tag))
    for num,word in enumerate(set_tag):
        print("num:%2s,word:%2s"%(num,word))
        dic_tag_id[word]=num



def load_data(voc_file):
  with open(voc_file, 'rb+') as load_data:
       word_id=pickle.load(load_data)
       return word_id

def data_corpus(corpus_path,word2id,label2id):
   train_tag=[]
   num=0
   with open(corpus_path,mode="r") as fr:
       for line  in  fr.readlines():
          data=line.strip().split()
          if data[0] in word2id:
            word=word2id[data[0]]
            tag=label2id[data[1]]
            train_tag.append([word,tag])
          else:
             num=num+1
             word=0
             tag=label2id[data[1]]
             train_tag.append([word, tag])
   return train_tag

def batch_yield(Trainx,labelY,sentence_length): # data中包含的是所有的数据
    """
    :param Trainx:
    :param labelY
    :param batch_size:

    :return:
    """
    num_samples=len(Trainx) #160399

    for i in range(sentence_length,num_samples,sentence_length):  #其实这个是sentence_length
        yield Trainx[i - sentence_length:i],labelY[i-sentence_length:i]

def sentence2id(word,vocab):
    sentence_id=[]
    sentence_id.append(vocab[word])
    return  sentence_id


def evalaute(logits,labels):
    """

    :param logits: [batchsize,seq_length]
    :param labels: [batchsize,seq_length]
    :return:
    """
    # print("logits length",len(logits)) #96  batchsize
    # print("logits[0] length",len(logits[0])) #64 sentence_length
    # print("labels length",len(labels)) #96 batchsize
    # print("labels[0] length",len(labels[0])) #64 sentence length
    entity_all=1
    entity_right=0
    entity_all_real=1
    for row in range(len(logits)):
        for num in range(len(logits[0])):
            if labels[row][num]>=1 and labels[row][num]<=8: #真实的实体个数
                entity_all_real=entity_all_real+1
            if logits[row][num]>=1 and logits[row][num]<=8:#预测出的实体个数
                     entity_all=entity_all+1
            if logits[row][num]==labels[row][num] and logits[row][num]!=0:
                    entity_right=entity_right+1
    accuracy=entity_right/entity_all #正确率
    recall=entity_right/entity_all_real #召回率
    #print("accuracy+recall+0.000001",accuracy+recall+0.000001)
    F1=2*accuracy*recall/float(accuracy+recall+0.000001) #F值
    print("accuracy:%5f recall:%5f F1:%5f"%(accuracy,recall,F1))



def pad_sequence(seqs,word_ids):
    print("in pad_seq")



if __name__=="__main__":
   data_corpus(src_file)
