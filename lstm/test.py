import pickle
import time
import numpy as np
import copy
from config import FLAGS
import json


def write_tag():
 lab=set();dic_word_id={}
 with open('data/train.txt','r+') as fr:
     for line in fr.readlines():
        data= line.strip().split()
        lab.add(data[1])

     for num,label in enumerate(lab):
         dic_word_id[label]=num

 with open('tag_voc.pickle', 'wb') as save_data:  #写入pickle文件中，字典
        pickle.dump(dic_word_id,save_data)


# logger = get_logger(log_path)  打log
# logger.info(dic_word_id)

def batch():
    print(3.2//2) #1.0  向下取整


def TestTime():
    #start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) #显示时间
    start_time=time.time()
    with open("vocab/train_voc.pickle",mode="rb") as fr:
          dic=pickle.load(fr)
    end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(start_time)
    print(end_time)
    print(time.time()-start_time) #测试当前方法花费的时间


def voc(voc_file):
    with open(voc_file, 'rb+') as load_data:
        word_id = pickle.load(load_data)
        return word_id

def key():
    tag2label= {'O': 0, 'I-ORG': 1, 'I-LOC': 2, 'I-PER': 3, 'I-MISC': 4, 'B-LOC': 5, 'B-PER': 6, 'B-MISC': 7, 'B-ORG': 8}
    label= "B-ORG"
    print(tag2label[label])

def len():
    a=['1','2','3','4']
    b=np.array(a)
    print(b.shape)

def shape():
    a=[]
    for i in range(10):
     a.append(i)
     b=np.array(a)
    print("b:",b.shape) #(10,)
    c=[[1,2,3]]
    c=np.array(c)
    print("c",c.shape) #(1, 3)
    d=np.ones(shape=[64,])*64
    print(d.shape) #(64,)

def numpy_test():
     # res=np.ones(shape=[64, ]) * 64
     # print("res",res)
     # res2=np.ones([96*64])
     # print("res2",res2)
     # print("res2_shape",res2.shape)
     score=np.ones([3,4,5])
     print("score[0]",score[0])
     print("score",score)
     #lengths = [len(score[0])]
     #print("length of a[0]",len(score[0]))

def ListCompare():
  a=[]
  b=[]
  for i in range(10):
      a.append(i)
  for j in range(10,20):
      b.append(j)

  print(a==b)

def copy_method():
    a=[1,2,3]
    b=a
    b.clear()
    print("a",a)
    print("b",b)

def list():
    a=[1,2,3]
    b=copy.copy(a)
    b.clear()
    print("a",a) #[1,2,3]
    print("b",b) #[]
    #a占用了一些内存地址，b中copy了一份


    c=[[4,5,6],[7,8,9],[10,11,12]]
    d=copy.copy(c) #copy的时候只是保存了最外层的地址
    print("c==d",c==d) #c,d是指向同一个地址的
    d[0][0]=10
    print("c",c)        #c [[10, 5, 6], [7, 8, 9], [10, 11, 12]],指向指针的指针改变了值
    print("d",d)        #d [[10, 5, 6], [7, 8, 9], [10, 11, 12]]
    d.clear()
    print("c",c)  #c [[10, 5, 6], [7, 8, 9], [10, 11, 12]]
    print("d",d)  #d []，清除了指向那个地址的指针




#在内存里，列表里只是存储了子列表的内存地址，子列表在内存里是单独存储的

if __name__=="__main__":
   #word_id =voc(FLAGS.voc_file)
   #numpy_test()
   #print("label_id",label_id)
   #write_tag()
   #key()
   #len()
   #shape()
   #ListCompare()
   #copy_method()
    list()