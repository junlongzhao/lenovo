import  json
import numpy as np
from SaveText import savetext
# def demo():
#   p0num=[1,2,3,2,1]
#   p1num=[2,3,4,5,2,1]
#   p0num=np.array(p0num)
#   p0denom=26.0
#   p1denom=21.0
#   p0list=[]
#   for i in range(len(p0num)):
#       p0list.append(p1denom)
#   print(p0list)
#   #p1denom=np.array(p1denom)
#   #p1vec = math.log(p1num /p1denom)
#   res=[math.log(p0/p) for p0,p in zip(p0num,p0list)]
#   print(res)

# def demo1():
#   a=[1,2,3,4]
#   b=[5,6,7,8]
#   c=[9,10,11,12]
#   for x,y,z in zip(a,b,c):
#     print("x",x)
#     print("y",y)
#     print("z",z)
#
#
# demo1()

# def readtext():
#   with open("news.txt",encoding="utf-8") as fr:
#     lines=fr.readlines()
#     for line in lines:
#       print(line)
# readtext()

# def add():
#  z = [[1,2,3]]
#  y = [[2, 3,4]]
#  print(type(z))
#  result=np.sum((z+y),axis=0)
#  print(result)
# add()

def readfile():
 with open("data.json",encoding="utf-8") as fr:
   lines=fr.readlines()
   num=0
   for line in lines:
       num=num+1
       #lines = json.loads(line)
       if num > 60 and num <=100:
         print(line)
         #savetext("test.json",line.strip())

if __name__=="__main__":
 readfile()