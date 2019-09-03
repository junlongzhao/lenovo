import jieba
import re
import pickle
import random
# def read_file(): #这种是数据量很大无法全部读入内存时的方法,是按条读取的
#    with open("data/TD.txt",'r',encoding="utf-8") as f:
#        line = f.readline().strip()
#        while line.strip():
#            print(line)
#            line = f.readline().strip()

def read_csv():
    id_list=[]
    content_list=[]
    with open("data/Test_DataSet.csv",'r',encoding='utf-8') as f:
          for num,line in enumerate(f.readlines()):
              content=line.strip().split(',',1)
              id_list.append(content[0])
              content_list.append(content[1])
          id_dealcontent(id_list,content_list)

def id_dealcontent(id_list,content_list):
  for id,content in zip(id_list,content_list):
      conttent_after=dealcontent(content)
      full_content=id+","+conttent_after
      savecontext("Test_id_content.csv",full_content)


def concatdata(ids,contents):
    data={}
    with open("data/Train_DataSet_Label.csv", 'r', encoding='utf-8') as f:
        for num,line in enumerate(f.readlines()):
            content=line.strip().split(',')
            data[str(content[0])]=content[1:] #key:id,value:label
    for id,content in zip(ids,contents):
       label=data[id]
       content_all=content+"__label__"+label[0]
       savecontext("Train.csv",content_all)

def savecontext(filename,contents):
    fh = open(filename, 'a', encoding='utf-8')
    fh.write(contents + "\n")
    fh.close()


def readfile():
    with open('Train.csv','r',encoding='utf-8') as fr:
      for line in fr.readlines():
        data=line.strip().split("__label__")
        content=data[0]
        label=data[1]
        result=dealcontent(content)
        full=result+"__label__"+label
        savecontext('Train_label.csv',full)

def dealcontent(content):
    text = ''.join(re.findall(u'[\u4e00-\u9fff]+', content))
    word = jieba.cut(text)
    full = " ".join(word)
    return  full


def dic():
    with open('data/Train__label.csv', 'r', encoding='utf-8') as fr:
        dic=set()
        word_id={}
        for line in fr.readlines():
            datalist=line.strip().split()
            for data in datalist:
             dic.add(data)
        for num,content in enumerate(dic):
            word_id[content]=num

    with open('word_id.pickle', 'wb') as save_data:  # 写入pickle文件中，字典
             pickle.dump(word_id, save_data)



# def train(): #去掉label
#     with open('data/Train_label.csv', 'r', encoding='utf-8') as fr:
#         for line in fr.readlines():
#             content=line.split("__label__")
#             savecontext('data/train.csv',content[0])

def help():
    a=['1','2','3','4','5']
    b=['6','7','8','9','10']
    Z=[]
    for x,y in zip(a,b):
        z=(x,y)
        Z.append(z)
    random.shuffle(Z)
    c,d=zip(*Z)
    print(c)
    print(d)






if  __name__=="__main__":
     read_csv()
    #readfile()
    #dic_label()
    #train()
     #Testdata()