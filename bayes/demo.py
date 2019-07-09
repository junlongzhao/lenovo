from  NaiveBayes import  NaiveBayesBase
from data import  loadDataset1
from data import loadDataset2
import numpy as np
import json
from SaveText import savetext

def train():
    #Train,label=loadDataset()
    #nb=NB()
    #vocab,voc_pos,voc_neg=nb.train_set(Train,label) #返回词典
    vocab,dic_pos,dic_neg=nb.train_set(Train,label)
    print("length of dic_neg",len(dic_neg))
    print("length of dic_pos",len(dic_pos))
    # for key,value in dic_pos.items():
    #     print("%s %d"%(key,value))
    DocMaxtri=nb.DocToMaxtri(Train,vocab)
    # p0,p1=nb.TrainNB(Train,label,dic_neg,dic_pos) #得到分子上的条件概率
    nb.TrainNB(Train,DocMaxtri,dic_neg,dic_pos)

def Testmethod():
   Train,label=loadDataset2() #此处是用已经训练好的数据集做相关参数的操作
   #Train = loadDataset1()
   nb=NaiveBayesBase()
   Train_voc=nb.train_voc(Train)
   print("Train_voc",Train_voc)
   Train_maxtri=nb.DocToMaxtri(Train,Train_voc) #这里才做成one-hot形式
   #print("Train_maxtri",Train_maxtri)
   print("in fit")
   nb.fit(Train_maxtri,label)
   print("out fit")
   #testEntry1 =[['love', 'my', 'dalmation'],   ['stop', 'posting', 'stupid', 'worthless', 'garbage']]
   #testEntry1 =[['中国', '军情', '新浪', '军事']]

   #testEntry1,label=loadDataset1()
   #testEntry1 = loadDataset1() #这里是传入待分类的数据，即预测集。
   testEntry1,id=LoadPredictdData()
   Test_maxtri=nb.DocToMaxtri(testEntry1,Train_voc)
   predict_result=nb.predict(Test_maxtri)
   for i in range(len(predict_result)):
       #print("第%d真实结果:%d"%(i,label[num]))
       print("第%d预测结果:%d"%(i,predict_result[i]))
   #return predict_result,id



def LoadPredictdData():
    all_content=[]
    all_id=[]
    with open("data.json", encoding="utf-8") as fr:
        for line in fr.readlines():
            content_all=json.loads(line)
            content=content_all['content']
            id=content_all['_id']
            con=content.split()
            all_content.append(con)
            all_id.append(id)
    return all_content,all_id

def readraw():
    with open("news2016zh_train.json",encoding="utf-8") as fr:
        for line in fr.readlines():
            print(line)


if __name__ =="__main__":
    #train()
   # Testmethod()
    #LoadPredictdData()
     readraw()

