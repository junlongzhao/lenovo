import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import random
UNK_ID=1


def create_vocabulary(file_path, vocab_size):
    voc_word2index = {}
    voc_index2word = {}
    voc_label2index = {}
    voc_index2label = {}
    file_object = open(file_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    c_inputs = Counter()
    C_inputs_label = Counter()
    for line in lines:
        raw = line.strip().split("__label__")
        input_list = raw[0].strip().split()
        input_label = raw[1:]
        c_inputs.update(input_list)
        C_inputs_label.update(input_label)
        vocab_list = c_inputs.most_common(vocab_size)
        label_list = C_inputs_label.most_common()
    for i, content in enumerate(vocab_list):
        word, _ = content
        voc_word2index[word] = i
        voc_index2word[i] = word
    for i, label_content in enumerate(label_list):
        label, _ = label_content
        print("label:", label)  # 2是为无效页面
        voc_label2index[label] = i
        voc_index2label[i] = label
    return voc_word2index, voc_index2word, voc_label2index, voc_index2label


def load_data_multiable(traing_data_path, vocabulary_word2index, voc_label2index, sentence_len, training_portion=0.9):
    TrainX = []
    TrainY = []
    label_size = len(voc_label2index)
    print("label_size:", label_size)
    file_objectwo = open(traing_data_path, mode='r', encoding='utf-8')
    lines = file_objectwo.readlines()
    numbers = len(lines)
    for line in lines:
        raw_list = line.strip().split("__label__")
        # print("type raw_list",type(raw_list))  #list
        input = raw_list[0].strip().split()
        input_label = raw_list[1:]
        x = [vocabulary_word2index.get(x, 1) for x in input]
        # print("x",x)在的就给编号，不在的就是1
        label = [voc_label2index[label] for label in input_label]
        # print("label",label)
        y = transform_mutilabel_as_multihot(label, label_size)
        TrainX.append(x)
        TrainY.append(y)
        # print("TrainX type before",type(TrainX))
    TrainX = pad_sequences(TrainX, maxlen=sentence_len, value=0.)  # 这个方法可以把list,变成numpy
    TrainY = np.array(TrainY)
    # print("TrainX type after ",type(TrainX))
    train_num = int(numbers * training_portion)
    train = (TrainX[0:train_num], TrainY[0:train_num])
    test = (TrainX[train_num + 1:], TrainY[train_num + 1:])
    return train, test


def load_data(training_data_path,word_id,training_portion=0.9):
    X = [];
    Y = [];
    with open(training_data_path, "r", encoding='utf-8') as fr:  # 这种按行读取的方式不会将数据全部加载进入内存中。
        for line in fr.readlines():
            data = line.strip().split("__label__")[:-1]
            labely = line.strip().split("__label__")[1]
            labely = int(labely)
            data_list = data[0].split()
            # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            # tokens = tokenizer.tokenize(data_text) #切成单个的字
            # x = tokenizer.convert_tokens_to_ids(tokens)
            x=[word_id.get(x,UNK_ID) for x in data_list]
            label = transform_mutilabel_as_multihot(labely, 3)
            X.append(x)
            Y.append(label)
        #print("length X",len(X)) #7335
        #print("length Y",len(Y)) #7335

        numbers = len(X)
        Train_Label=[]
        TrainX = pad_sequences(X, maxlen=400, value=0.)

        for x,y in zip(TrainX,Y):
            train=(x,y)
            Train_Label.append(train)
        random.shuffle(Train_Label)
        train_num = int(numbers * training_portion)
        train = Train_Label[0:train_num]
        test =Train_Label[train_num + 1:]
        return train, test

def load_data_predict(predict_path,word_id):
    X=[];Y=[]
    with open(predict_path,'r',encoding='utf-8') as fr:
        for line in fr.readlines():
           raw=line.strip().split(',')
           id=raw[0]
           content=raw[1:]
           content_list=content[0].split(" ")
           X_id=[word_id.get(x,UNK_ID) for x in content_list]
           X.append(X_id)
           Y.append(id)
        TrainX = pad_sequences(X, maxlen=400, value=0.)
        #print("length Trainx",len(TrainX)) # 7356
        #print("length Y",len(Y)) # 7356
        print("Y",Y)
        return TrainX,Y


def  transform_mutilabel_as_multihot(label,label_size):
    result=np.zeros(label_size)
    result[label]=1
    return result
