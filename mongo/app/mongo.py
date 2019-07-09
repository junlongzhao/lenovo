import pymongo
import json
def connect():
    client=pymongo.MongoClient('localhost',27017) #建立数据库连接，指定ip和端口号
    mydb = client.mydb #指定mydb数据库
    collection = mydb.user #指定mydb数据库里user集合
    data3={"age":12,"name":"laozhao","school":"UESTC","label":"null"}
    data4={"_id":1,"age": 23, "userName": "laozhao"}
    data5={"_id":2,"age": 23, "userName": "zhangsan","school":"liuzhong"}
    data6_before = {"_id": 16, "age": 23, "userName": "zhangsan", "school": "None"} #之前
    data6={"_id":0,"age": 23, "userName": "zhangsan","school":"qizhong"} #之后
    print(type(data5))
    data7={"_id":0,"content":"中国 军情 新浪 军事","label":"pos"}
    #collection.insert_one(data6)
    #collection.update_one({ "_id": 0 },{"$set":data7}) #这是更新的方法，必须保证id值是相同的，且字段要相同
    # 查询内容
    print(collection.find_one({ "_id": 0 }))

def ReadContent():
    client = pymongo.MongoClient('localhost', 27017)  # 建立数据库连接，指定ip和端口号
    mydb = client.mydb  # 指定mydb数据库
    collection = mydb.user  # 指定mydb数据库里user集合
    #with open("data/data.json",encoding="utf-8") as fr:
    with open("data/data.json",encoding="utf-8") as fr:
        for line in fr.readlines():
            data=json.loads(line) #可将内容由字符串转换成字典
            collection.insert_one(data)

if __name__=="__main__":
  connect()
  #ReadContent()