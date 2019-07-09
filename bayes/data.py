import  numpy as np
import json
def loadDataset2():
    postionglist=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please','my'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid','rubissh','unknown'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage',],
                 ['mr', 'licks', 'ace', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid','hate',]]
    classdevc=[0,1,0,1,0,1]
    for i in range(len(classdevc)):
        print(type(i))
    return postionglist,classdevc

def loadDataset1():
        with open("test.json", encoding="utf-8") as fr:
            lines = fr.readlines()
            postionglist=[]
            classdevc=[]
            for line in lines:
                  data=json.loads(line)
                  content=data['content']
                  label=data['label']
                  #label=int(label)
                  content_list=content.split( )
                  postionglist.append(content_list)
                  #classdevc.append(label)
            return postionglist#,classdevc

def loadDataset2():
    with open("raw.json", encoding="utf-8") as fr:
        lines = fr.readlines()
        postionglist = []
        classdevc = []
        for line in lines:
            data = json.loads(line)
            content = data['content']
            label = data['label']
            label=int(label)
            content_list = content.split()
            postionglist.append(content_list)
            classdevc.append(label)
        return postionglist ,classdevc


if __name__ =="__main__":
    loadDataset1()
    #loadDataset()