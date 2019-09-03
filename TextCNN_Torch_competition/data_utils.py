import numpy as np
import pickle
import os
from config import args
from pytorch_transformers import *
import random
UNK_ID=1
def batch_helper(TrainX,TrainY,batch_size): #data_shape[num_samples,sentence_length]
    data_size = len(TrainX)
    num_batches_per_epoch = int((len(TrainX) - 1) / batch_size) + 1
    #print("num_batches_per_epoch", num_batches_per_epoch)# 10764
    for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield TrainX[start_index:end_index],TrainY[start_index:end_index]

def batch_helper_predict(test,batchsize):
    data_size=len(test)
    num_batches=int((len(test)-1)/batchsize)+1
    for batch_num in range(num_batches):
        start=batch_num*batchsize
        end=min((batch_num + 1) * batchsize, data_size)
        yield test[start:end]


def load_data(training_data_path,word_id,label_id,training_portion=0.9):
    X=[];Y=[];
    with open(training_data_path,"r",encoding='utf-8') as fr: #这种按行读取的方式不会将数据全部加载进入内存中。
        for line in fr.readlines():
            data=line.strip().split("__label__")[:-1]
            label = line.strip().split("__label__")[-1:]
            label =label[-1]
            data_list=data[0].split()
            y=[label_id[label]]
            x=[word_id.get(x,UNK_ID) for x in data_list]
            X.append(x)
            Y.append(y)
        numbers = len(X)
        TrainX=pad_sequences(X,maxlen=20,value=0.)
        TrainY=np.array(Y)
        #print("X length:",len(X)) #765376
        #print("Y length:",len(Y)) #765376
        train_num = int(numbers * training_portion)
        train = (TrainX[0:train_num], TrainY[0:train_num])
        test = (TrainX[train_num + 1:], TrainY[train_num + 1:])
        return train, test




def load_data1(training_data_path,word_id,training_portion=0.9):
    X = [];
    Y = [];
    with open(training_data_path, "r", encoding='utf-8') as fr:  # 这种按行读取的方式不会将数据全部加载进入内存中。
        for line in fr.readlines():
            data = line.strip().split("__label__")[:-1]
            label = line.strip().split("__label__")[-1:]
            label = float(label[-1])
            data_list = data[0].split()
            # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            # tokens = tokenizer.tokenize(data_text) #切成单个的字
            # x = tokenizer.convert_tokens_to_ids(tokens)
            #print("ids",x)
            x=[word_id.get(x,UNK_ID) for x in data_list]
            X.append(x)
            Y.append(label)
        #print("length X",len(X)) 7355
        #print("length Y",len(Y)) 7355
        numbers = len(X)
        Train_Label=[]
        TrainX = pad_sequences(X, maxlen=400, value=0.)
        TrainY = np.array(Y)
        for x,y in zip(TrainX,TrainY):
            train=(x,y)
            Train_Label.append(train)
        random.shuffle(Train_Label)
        train_num = int(numbers * training_portion)
        train = Train_Label[0:train_num]
        test =Train_Label[train_num + 1:]
        return train, test

def load_data_bert(training_data_path,word_id,training_portion=0.9):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    X=[];Y=[]
    with open(training_data_path,'r',encoding='utf-8') as fr:
        for line in fr.readlines():
            data = line.strip().split("__label__")[:-1]
            label = line.strip().split("__label__")[-1:]
            label = float(label[-1])
            data_str=data[0]
            tokens = tokenizer.tokenize(data_str)  # 把句子切割成字
            ids = tokenizer.convert_tokens_to_ids(tokens)
            X.append(ids)
            Y.append(label)
        numbers = len(X)
        Train_Label=[]
        TrainX = pad_sequences(X, maxlen=400, value=0.)
        TrainY = np.array(Y)
        for x,y in zip(TrainX,TrainY):
            train=(x,y)
            Train_Label.append(train)
        random.shuffle(Train_Label)
        train_num = int(numbers * training_portion)
        train = Train_Label[0:train_num]
        test =Train_Label[train_num + 1:]
        return train, test

def load_data_test(test_data_path,word_id):
    X=[];ID=[]
    with open(test_data_path,'r',encoding='utf-8') as fr:
        for line in fr.readlines():
            id = line.strip().split(",")[0]
            content = line.strip().split(",")[-1:]
            data_list=content[0].strip().split()
            x = [word_id.get(x, UNK_ID) for x in data_list]
            X.append(x)
            ID.append(id)
        TestX = pad_sequences(X, maxlen=400, value=0.)
        return TestX,ID
            # Y.append(label)
        #     label = float(label[-1])
        #     data_str = data[0]
        #     X.append(ids)
        #     Y.append(label)
        # numbers = len(X)
        # Train_Label = []

        # TrainY = np.array(Y)


def create_vocab(training_data_path,cache_path,Judge=False):
    if os.path.exists(cache_path):
        print("load exit vocab")
        with open(cache_path, 'rb') as data_f:
            if Judge:
              dict=pickle.load(data_f)
              args.vocab_size=len(dict)
              return dict
            else:
                return pickle.load(data_f)
    else:
        print("create vocab")
        with open(training_data_path,'r',encoding='utf-8') as fr:
            dict=set()
            word_id={}
            line = fr.readline().strip()
            while line:
              data = line.strip().split("__label__")[:-1]
              data_list = data[0].split()
              for word in data_list:
                  dict.add(word)
              line = fr.readline().strip()
        args.vocab_size = len(dict)
        for num,data in enumerate(dict):
             word_id[data]=num
        with open(cache_path,'wb') as fr:
                pickle.dump(word_id,fr)
        return word_id

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences.

    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).

    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)

    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def create_vocab_tag(training_data_path,cache_path): #创建label_id,只是在label_id的时候调用一下而已
    if os.path.exists(cache_path):
        print("load exit vocab")
        with open(cache_path, 'rb') as data_f:
            return pickle.load(data_f)
    else:
        with open(training_data_path, 'r', encoding='utf-8') as fr:
              dict_tag=set()
              label_id={}
              line=fr.readline()
              while line:
                  label=line.strip().split("__label__")[-1:]
                  label=label[-1]
                  print(type(label))
                  dict_tag.add(label)
                  line = fr.readline().strip()
              for num,label in enumerate(dict_tag):
                   label_id[label] = num
              with open(cache_path, 'wb') as fr:
                  pickle.dump(label_id, fr)
              return label_id


if __name__=="__main__":

    # data=[]
    # num=0
    # data = np.arange(1000).reshape(100, 10)
    # batches=batch_helper(data,10)
    # for batch in batches:
    #     print(batch)
    #load_data()
    #word_id=create_vocab(args.training_data_path,args.vocab_path)

   # label_id=create_vocab_tag(args.training_data_path,args.vocab_tag_path)
   # print("label_id",label_id)
    print("oz")