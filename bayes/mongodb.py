import pymongo
import json
from demo import Testmethod
#-*-coding:utf-8-*-
def connect(id): #查询操作
    client=pymongo.MongoClient('localhost',27017) #建立数据库连接，指定ip和端口号
    mydb = client.mydb #指定mydb数据库
    collection = mydb.user #指定mydb数据库里user集合
    data6_before = {"_id": 16, "age": 23, "userName": "zhangsan", "school": "None"} #之前
    data6={"_id":0,"age": 23, "userName": "zhangsan","school":"qizhong"} #之后
    #collection.insert_one(data6)
    #collection.update_one({ "_id": 0 },{"$set":data7}) #这是更新的方法，必须保证id值是相同的，且字段要相同
    # 查询内容
    qurey_content=collection.find_one({ "_id": id }) #这个是字典，有多个键组成。
    res=json.dumps(qurey_content,ensure_ascii=False) #内容由字典转化成字符串
                                                    # ensure_ascii，这个参数非常重要，为False时是，utf-8码
    return res

def findContent(id): #查询操作
    client = pymongo.MongoClient('localhost', 27017)  # 建立数据库连接，指定ip和端口号
    mydb = client.mydb  # 指定mydb数据库
    collection = mydb.user  # 指定mydb数据库里user集合
    accuracy_list=[]
    mongo_accuracy_list=[]
    accuracy=0
    # with open("accuracy.txt",encoding="utf-8") as fr:
    #     for line in fr.readlines():
    #         accuracy_list.append(line)    此处是测试正确率
    #     print(len(accuracy_list))
    # for id in range(60,100):
    #   qurey_content = collection.find_one({"_id": id})
    #   mongo_accuracy_list.append(qurey_content['label']+"\n")
    # print(len(mongo_accuracy_list))
    qurey_content = collection.find_one({"_id": id})
    print(qurey_content)

def update(): #更新操作
    client = pymongo.MongoClient('localhost', 27017)  # 建立数据库连接，指定ip和端口号
    mydb = client.mydb  # 指定mydb数据库
    collection = mydb.user  # 指定mydb数据库里user集合
    #qurey_content = collection.find_one({"_id": id})
    predict_result,id=Testmethod()
    #print("predict_result_length",len(predict_result))
    #print("id_length",len(id))
    for i  in range(100,922):
      print("i:",i)#这个i是预测第i次的结果
      qurey_content = collection.find_one({"_id":i})
      #print("predict_result:%d,qurey_content_id:%d"%(int(predict_result[i]),int(qurey_content['label'])))
      print("qurey_content_label_before",qurey_content['label'])
     # print("predict_result",predict_result[i])
      if i==qurey_content['_id']: #这是字段更新的方法
        qurey_content['label']=predict_result[i]
        print("qurey_content_label_after",qurey_content['label'])
        collection.update_one({"_id": i}, {"$set": qurey_content})

if __name__=="__main__":
   update()