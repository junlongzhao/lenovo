import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter


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


def load_data(content, word2index):
    print("content type:", type(content))
    content_list = content.split()
    print("content_list", content_list)
    x = [word2index.get(x, 1) for x in content_list]  # 存在就转化成index，不存在就是1
    print("load_data")
    return x



def  transform_mutilabel_as_multihot(label,label_size):
    result=np.zeros(label_size)
    for i in label:
        result[i]=1
    return result
