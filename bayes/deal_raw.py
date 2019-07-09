import jieba
import re
def readfile():
 with open("news.txt",encoding="utf-8") as fr:
   lines=fr.readlines()
   num=0
   for line in lines:
       print("line",line)
       num=num+1
       text = ''.join(re.findall(u'[\u4e00-\u9fff]+', line))
       word = jieba.cut(text)
       full = " ".join(word)
       print("line:%d %s "%(num,full))



if __name__=="__main__":
    readfile()
