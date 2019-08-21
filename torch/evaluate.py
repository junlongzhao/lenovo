import torch
def eva(predict,label,batchsize):
    result=torch.argmax(predict,dim=-1)
    label=label.numpy()  #将tensor转化成numpy
    result=result.numpy()
    accuracy_all=0
    for num in range(len(result)):
         print("result",result)
         print("label",label)
         if result[num]==label[num]:
             accuracy_all=accuracy_all+1
    print("accuracy:",accuracy_all/batchsize)
