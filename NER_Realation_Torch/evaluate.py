import torch
# def evalution(logits,labels):  #NER
#     """
#     :param logits: [batchsize,seq_length]
#     :param labels: [batchsize,seq_length]
#     :return:
#     """
#     # print("logits length",len(logits)) #96  batchsize
#     # print("logits[0] length",len(logits[0])) #64 sentence_length
#     # print("labels length",len(labels)) #96 batchsize
#     # print("labels[0] length",len(labels[0])) #64 sentence length
#     entity_all=1
#     entity_right=0
#     entity_all_real=1
#     for row in range(len(logits)):
#         for num in range(len(logits[0])):
#             if labels[row][num]>=1 and labels[row][num]<=8: #真实的实体个数
#                 entity_all_real=entity_all_real+1
#             if logits[row][num]>=1 and logits[row][num]<=8:#预测出的实体个数
#                      entity_all=entity_all+1
#             if logits[row][num]==labels[row][num] and logits[row][num]!=0:
#                     entity_right=entity_right+1
#     accuracy=entity_right/entity_all #正确率
#     recall=entity_right/entity_all_real #召回率
#     #print("accuracy+recall+0.000001",accuracy+recall+0.000001)
#     F1=2*accuracy*recall/float(accuracy+recall+0.000001) #F值
#     print("accuracy:%5f recall:%5f F1:%5f"%(accuracy,recall,F1))

def evaluate_classify(logits,label):
    accuracy_num=0
    for i in range(len(logits)):
        if logits[i]==label[i]:
            accuracy_num=accuracy_num+1
    print("accuracy",accuracy_num/len(logits))

def predict_all(dev_iter,cnn):
    predictall=[];labelall=[];
    accuracy_num=0;allnum=0;#计算所有的label个数
    for step,batch in enumerate(dev_iter):
        content=batch.content
        label=batch.label
        logits=cnn(content)
        outputs = torch.argmax(logits, dim=-1)
        logits=torch.Tensor.long(outputs)
        labelall.append(label)
        predictall.append(logits)
    for row in range(len(labelall)):
        for num in range(len(labelall[row])):
            allnum=allnum+1
            # print("type label",type(labelall[row][num]))
            # print("type logit",type(predictall[row][num]))
            # print(labelall[row][num])
            # print(predictall[row][num])
            if labelall[row][num].numpy()==predictall[row][num].numpy():
              accuracy_num=accuracy_num+1
    print("allnum", allnum)
    print("predict accuracy",accuracy_num/allnum)