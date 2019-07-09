#工具类
# 传入的应该是文本做成的one-hot矩阵
import math
import numpy as np
def list_sum(sentence_length,do_list,voc):

    """
    :param do_list: 此处是传入的是每一个文本的矩阵形式  比如：['my', 'dog', 'has', 'flea', 'problems', 'help', 'please', 'my']
    :return:返回矩阵形式 [2. 1. 1. 1. 1. 1. 1. 0.],2代表词语出现的频数
    """
    print("词典的长度",len(voc))
    # for key,value in voc.items():
    #     print("voc %s %d"%(key,value))
    input=np.ones(sentence_length)
    for word in do_list:
         if(word in voc):
            #print("word in voc num",do_list.index(word)) #获取一个词语在List中下标的方法
            input[do_list.index(word)]=input[do_list.index(word)]+1
    print("input",input)
    return input

def log(p0num,p1):
    """
    :param p0num: 被除数 numpy， 数据的one-hot形式
    :param p1: 除数    单个数字,即词典的长度
    :return:  被除数/除数
    """
    numlist=[]
    p0list = []
    for num in p0num:
       if num!=0:
           numlist.append(num)
    for i in range(len(numlist)):
        p0list.append(p1)
    print("numlist",numlist)
    print("p0list",p0list)
    res = [math.log(p0 / p) for p0, p in zip(p0num, p0list)] #真数不能够为0
    # res_sum=np.sum(res)
    # print("res_sum",res_sum)
    return res

def sum(maxtri):
   #print("maxtri",maxtri)
   number=0
   for word in range(len(maxtri)):
       if maxtri[word]==1:
           number=number+1
   return number