import numpy as np
from imp import sum
class NaiveBayesBase(object):

    def __init__(self):
        pass

    def train_voc(self,train_set): # 建立词典
        self.docnumber = len(train_set)
        temp_set = set()
        [temp_set.add(word) for doc in train_set for word in doc]
        self.vocab = list(temp_set)
        self.vocablen = len(self.vocab)
        self.MaxSentence=self.vocablen
        # for key, value in dic_pos.items():
        #     print("%s %d" % (key, value))
        return self.vocab


    def DocToMaxtri(self,train_set,vocab):
       docvec=np.zeros([len(train_set),self.MaxSentence])
       for indx in range(len(train_set)):
         for word in train_set[indx]:    #重复出现不管暂时
             if word in vocab:
                 docvec[indx,vocab.index(word)]=1
             # else:
             #     docvec[indx,vocab.index(word)]=0
       return docvec

    def fit(self, trainMatrix, trainCategory): #训练的时候用来初始化获得上面的条件概率
        """
        :param trainMatrix: 训练矩阵，即向量化表示后的文档（词条集合）
        :param trainCategory: 文档中每个词条的列表标注
        :return:  p0Vect :属于0类别的概率向量(p(w1|C0),p(w2|C0),...,p(wn|C0))
                  p1Vect属于1类别的概率向量(p(w1|C1),p(w2|C1),...,p(wn|C1))
                  pAbusive : 属于1类别文档的概率
        """
        NumWords = len(trainMatrix[0])
        print("trainMatrix[0]",trainMatrix[0])
        print("NumWords",NumWords)
        NumDoc = len(trainMatrix)
        self.pAbusive=np.sum(trainCategory)/float(NumDoc)
        p0=np.ones(NumWords)
        p1=np.ones(NumWords)
        p0doc = 2.0 #下面初始化为2
        p1doc = 2.0
        for doc_number in range(len(trainMatrix)):
            if trainCategory[doc_number]==0:
               p0+=trainMatrix[doc_number] #list和numpy相加
               p0doc=p0doc+sum(trainMatrix[doc_number]) #统计正向情感中出现的所有词语总和
            else:
                p1+=trainMatrix[doc_number]
                p1doc=p1doc+sum(trainMatrix[doc_number])
        self.p1Vect = np.log(p1 / p1doc)   #这里是在计算分子，为了保证不溢出，用log运算
        self.p0Vect = np.log(p0 / p0doc)
        return self

    def predict(self, testX):
        '''
        朴素贝叶斯分类器
        Args:
            testX : 待分类的文档向量（已转换成array）
            p0Vect : p(w|C0)
            p1Vect : p(w|C1)
            pAbusive : p(C1)
        Return:
            1 : 为侮辱性文档 (基于当前文档的p(w|C1)*p(C1)=log(基于当前文档的p(w|C1))+log(p(C1)))
            0 : 非侮辱性文档 (基于当前文档的p(w|C0)*p(C0)=log(基于当前文档的p(w|C0))+log(p(C0)))
        '''
        result=[]
        for number in range(len(testX)):
          p1 = np.sum(testX[number] * self.p1Vect) + np.log(self.pAbusive)
          p0 = np.sum(testX[number] * self.p0Vect) + np.log(1 - self.pAbusive)
          if p1 > p0:
              result.append(1)
          else:
              result.append(0)
        return result

