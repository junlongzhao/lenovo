import torch
def eva(predict,label,batchsize):
    result=torch.argmax(predict,dim=-1)
    label=label.cuda() #将tensor转化成numpy
    result=result
    accuracy_all=0
    for num in range(len(result)):
         if result[num]==label[num]:
             accuracy_all=accuracy_all+1
    print("train accuracy:",accuracy_all/len(predict))

def eva_test(predict,label,batchsize):
    result=torch.argmax(predict,dim=-1)
    label=label.cuda() #将tensor转化成numpy
    result=result
    accuracy_all= 0

    for num in range(len(result)):
         if result[num]==label[num]:
             accuracy_all=accuracy_all+1
    print("test accuracy:",accuracy_all/len(predict))