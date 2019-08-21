import torch
import numpy as np
from evaluate import eva
from  model import EncoderRNNWithVector
def _test_rnn_rand_vec():

    # 这里随机生成一个 Tensor，维度是 1000 x 10 x 200；其实就是1000个句子，每个句子里面有10个词向量，每个词向量 200 维度，其中的值符合 NORMAL 分布。

    _xs = torch.randn(1000, 10, 200)
    _ys = []

    # 标签值 0 - 5 闭区间
    for i in range(1000):
        _ys.append(1)

    # 隐层 200，输出 6，隐层用词向量的宽度，输出用标签的值得个数 （one-hot)
    encoder_test = EncoderRNNWithVector(200, 6)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(encoder_test.parameters(), lr=0.001, momentum=0.9)

    num_data=len(_xs) #1000
    batchsize=20
    num_epoches=10
    for epoch in range(num_epoches):
        for start, end in zip(range(0,num_data,batchsize),range(batchsize,num_data,batchsize)):
            encoder_hidden = encoder_test.init_hidden()
            input_data = torch.autograd.Variable(_xs[start:end])
            output_labels = torch.autograd.Variable(torch.LongTensor(np.array([_ys[start:end]])).reshape(batchsize))#output_labels需要为LongTensor
            encoder_outputs, encoder_hidden = encoder_test(input_data, encoder_hidden) #此处调用前向传播

            optimizer.zero_grad()
            predict = encoder_outputs.view(batchsize,-1)

           # print("predict_shape",predict.size()) #predict_shape torch.Size([20, 6])
           # print("output_labels", output_labels.size()) #output_labels torch.Size([20])


            loss = criterion(predict, output_labels)

            loss.backward()
            optimizer.step()

            eva(predict,output_labels,batchsize)

    return

if __name__=="__main__":
     _test_rnn_rand_vec()