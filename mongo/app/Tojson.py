import json
from save import savetext
# -*- coding: utf-8 -*-
def Tojson():
    with open("raw.txt",encoding="utf-8") as fr:
        lines=fr.readlines()
        num=0;dict={}
        for line in lines:
          num=num+1
          dict[num]=line
          content=json.dumps(dict,ensure_ascii=False)
          with open('test_data.json', 'a',encoding="utf-8") as json_file:
              json_file.write(content)

    for key,value in dict.items():
        print("%d %s"%(key,value))
        json_str = json.loads(dict)
        print(dict)

def to():
  data1 = {'0' : '中国'}
  data2={'1' : '北京'}
  data3={'2':"美国"}
  content=json.dumps(data3,ensure_ascii=False)
  content.encode('utf-8').decode('unicode_escape')
  with open('test_data1.json', 'a',encoding="utf-8") as json_file:
      json_file.write(content+"\n")

def read():
    with open('id_raw.txt', 'r',encoding="utf-8") as fr:
      # load_dict = json.load(fr)
          num=0
          for line in fr.readlines():
              #line=json.loads(line)
              raw=line.split("__content__")
              content="{"+"\"_id\""+":"+raw[0]+","+"\"content\""+":"+"\""+raw[1]+"\""+","+"\"label\""+":"+"\""+raw[2].strip()+"\""+"}"
              print(content)
              raw=json.dumps(content,ensure_ascii=False)
              raw.encode('utf-8').decode('unicode_escape')
              with open('data.json', 'a', encoding="utf-8") as json_file:
                  json_file.write(content + "\n")


def IdtoContent():
    with open("raw.txt", encoding="utf-8") as fr:
        lines = fr.readlines()
        num=0
        for line in lines:
           content=str(num)+"__content__"+line.strip()+"__content__None"
           num=num+1
           savetext("id_raw.txt",content)



if __name__=="__main__":
    read()
    #to()
    #Tojson()
    #IdtoContent()