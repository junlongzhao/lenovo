import  numpy as np
from imp import  list_sum,log
class NB:
    def __init__(self):
        self.vocab=[] #词典
        self.idf=0  #词典的idf权值向量
        self.tf=0  #训练集的权值矩阵
        self.tdm=0  # P(x|yi)
        self.Pcates={} # P(yi)--是个类别字典
        self.labels = []  # 对应每个文本的分类，是个外部导入的列表，事先标注好的语料
        self.docnumber = 0  # 训练集文本数
        self.vocablen = 0  # 词典词长
        self.testset = 0   #  测试集
        self.MaxSentence=0

    def train_set(self,train_set,classdevc): #建立词典
        self.cate_prob(classdevc)
        self.docnumber=len(train_set)
        temp_set=set()
        [temp_set.add(word)  for doc in train_set for word in doc]
        self.vocab=list(temp_set)
        self.vocablen=len(self.vocab)
        self.voc_freq(train_set)    # 计算词频数据集
        # self.build_tmd()
        #self.DocToMaxtri(train_set,self.vocab)
        temp_set_pos=set()
        temp_set_neg=set()
        dic_neg = {}
        dic_pos={}
        print("classdevc",classdevc)
        for num,label in enumerate(classdevc):
            if label==0: #建立正向文本中的词典
              [temp_set_pos.add(word) for word in train_set[num]]
              voc_pos=list(temp_set_pos)
            else: # 建立负向文本中的词典
                [temp_set_neg.add(word) for word in train_set[num]]
                voc_neg=list(temp_set_neg)
        for num_neg,word_neg in enumerate(voc_neg):
           dic_neg[word_neg]=num_neg

        for num_pos,word_pos in enumerate(voc_pos):
            dic_pos[word_pos]=num_pos

        # for key, value in dic_pos.items():
        #     print("%s %d" % (key, value))

        # for key,value in dic_neg.items():
        #     print("%s %d"%(key,value))
        return self.vocab,dic_pos,dic_neg

    def cate_prob(self,classdevc):
        sum = 0
        for i in range(len(classdevc)):   #建立每一句对应的概率Py_i
            if classdevc[i] == 1:
                sum = sum + 1
        for j in range(len(classdevc)):
            self.Pcates[j] = classdevc[j] /sum   #就是一个字典，键为序列号，值为label在所有占比列。

    def voc_freq(self,train_set):
        self.idf=np.zeros([1,len(self.vocab)])  #1,词典数
        self.tf=np.zeros([self.docnumber,self.vocablen])  # 训练集文件数*词典数
        for indx in range(self.docnumber):  # 遍历所有的文本
            for word in train_set[indx]:  # 遍历文本中的每个词
                # print(self.vocab.index(word)) #list也可以 .index(word)
                self.tf[indx,self.vocab.index(word)]+=1 #每一个单词在一个文本中出现的次数
            for singleword in train_set[indx]:
                self.idf[0,self.vocab.index(singleword)]+=1 #每一个单词在所有文本中出现的次数

    # def build_tmd(self):
    #    self.tdm=np.zeros([len(self.Pcates),self.vocablen]) #类别个数*词典长度
    #    sumlist=np.zeros([len(self.Pcates),1])   # 统计每个分类的总值
    #    for  indx in range(self.docnumber):
    #        self.tdm[self.labels[indx]]+=self.tf[indx] #将同一类别的词向量空间相加，tf是一个二维矩阵
    #        sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
    #        self.tdm = self.tdm / sumlist
    #        print("oz")

    def TrainNB(self,Train,DoctoMaxtri,voc_neg,voc_pos):
        print("训练集长度",len(Train))
        voc_neg_length=len(voc_neg)
        voc_pos_length=len(voc_pos)
        p0veclist=[]
        p1veclist=[]
        for num in range(self.docnumber):
               p1num=list_sum(self.MaxSentence,Train[num],voc_neg) #[0. 1. 0. 0. 0. 0. 0. 0.]
               p0num=list_sum(self.MaxSentence,Train[num],voc_pos) #[2. 1. 1. 1. 1. 1. 1. 0.]
               p0vec=log(p0num,voc_pos_length)
               p1vec = log(p1num, voc_neg_length)
               p0veclist.append(p0vec)
               p1veclist.append(p1vec)
        print("p0vec",p0vec)
        print("p1vec",p1vec)
        print("p0vlist length",len(p0veclist))
        print("p1vlist lenghth",len(p1veclist))
        self.classfyNB(DoctoMaxtri,p0veclist,p1veclist)
        return p0veclist,p1veclist     #条件概率计算上面的分子

    def classfyNB(self,DoctoMaxtri,p0veclist,p1veclist):
        for vec,p0vec,p1vec in zip(DoctoMaxtri,p0veclist,p1veclist):
          p0=np.sum(p0vec*vec)+np.log(0.5)
          p1=np.sum(p1vec*vec)+np.log(0.5)
          print("p0",p0)
          print("p1",p1)
          if p0>p1:
              print("正向情感")
          else:
              print("负向情感")


    def DocToMaxtri(self,train_set,vocab):
       docvec=np.zeros([len(train_set),self.MaxSentence])
       for indx in range(self.docnumber):
         for word in train_set[indx]:
           if word in vocab:  #判断词语是否在list中
             docvec[indx,train_set[indx].index(word)]=1
           else:
             docvec[indx, train_set[indx].index(word)]=0
       # for indx in range(self.docnumber):
       #     print(docvec[indx])
       return docvec

    def predict(self,test):
          print("oz")