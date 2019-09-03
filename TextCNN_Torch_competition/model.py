import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_transformers import *

class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        Vocab = args.vocab_size  ## 已知词的数量
        Dim = args.embed_dim  ##每个词向量长度
        Cla = args.class_num  ##类别数
        Ci = 1  ##输入的channel数
        Knum = args.kernel_num  ## 每种卷积核的数量
        Ks = args.kernel_sizes  ## 卷积核list，形如[2,3,4]
        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        #self.model = BertModel.from_pretrained('bert-base-chinese')
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        x = self.embed(x)  # (N,W,D) (batchsize,sentence_length,embedding_size)
        #outputs = self.model(x)
        #x = outputs[0]
        #print("x.size()",x.size())
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        # print("x[0] shape",x[0].size()) #shape torch.Size([64, 100, 18])
        # print("x[1] shape",x[1].size()) #shape torch.Size([64, 100, 17])
        # print("x[2] shape",x[2].size()) # shape torch.Size([64, 100,16])
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum) 池化层

        x = torch.cat(x, 1)  # (N,Knum*len(Ks)) ([64, 300])
        x = self.dropout(x)
        logit = self.fc(x)
        #print("logit size",logit.size()) #[batchsize,num_classes]
        return logit

