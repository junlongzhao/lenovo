import torch
import model
from data_utils import create_vocab,load_data,batch_helper,load_data1,load_data_bert,load_data_test,batch_helper_predict
from config import args
import random
from evaluate import eva,eva_test
from pytorch_transformers import *

def main():
    word_id=create_vocab(args.training_data_path,args.vocab_path,True)
    #label_id=create_vocab(args.training_data_path,args.vocab_tag_path)
    args.class_num=3
    #train,test=load_data(args.training_data_path,word_id,label_id)
    train1,test1=load_data1(args.training_data_path,word_id)
    #train1,test1=load_data_bert(args.training_data_path,word_id)
    TrainX,TrainY=zip(*train1)
    testX,testY=zip(*test1)
    cnn=model.CNN_Text(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    opt_Adam = torch.optim.Adam(cnn.parameters(), lr=args.lr, betas=(0.9, 0.99))

    for epoch in range(1,args.epoches):
        print("epoch",epoch)
        batch_iter=batch_helper(TrainX,TrainY,args.batch_size)
        for trainx,trainy in batch_iter:
            #print("trainy length",len(trainy)) #batchsize
            input_data = torch.autograd.Variable(torch.LongTensor(trainx)).cuda()

            output_labels=torch.autograd.Variable(torch.LongTensor(trainy)).cuda()
            output_labels=output_labels. squeeze()
            #print("vocab_size",args.vocab_size)
            cnn_outputs=cnn(input_data)
            torch.save(cnn.state_dict(),args.parameters_path)
            opt_Adam.zero_grad()
            loss = criterion(cnn_outputs, output_labels)
            loss.backward()
            opt_Adam.step()
            # for param_tensor in cnn.state_dict():
            #     print(param_tensor, "\t", cnn.state_dict()[param_tensor].size())
            # for var_name in opt_Adam.state_dict():
            #     print(var_name, "\t", opt_Adam.state_dict()[var_name])
            eva(cnn_outputs,output_labels,args.batch_size)
        torch.save(cnn.state_dict(), args.parameters_path)
        run_val(testX,testY,cnn)

def run_val(testX,testY,cnn):
    batch_iter=batch_helper(testX,testY,args.batch_size)
    for testX,testY in batch_iter:
        input_data = torch.autograd.Variable(torch.LongTensor(testX)).cuda()
        output_labels = torch.autograd.Variable(torch.LongTensor(testY)).cuda()
        cnn_outputs = cnn(input_data)
        eva_test(cnn_outputs,output_labels,args.batch_size)

def predict():
    word_id = create_vocab(args.test_data_path, args.vocab_path, True)
    Test, ID= load_data_test(args.test_data_path, word_id)
    batch_iter = batch_helper_predict(Test, args.batch_size)
    cnn=torch.load(args.parameters_path)
    with torch.no_grad():
      for predict in batch_iter:
          input_data = torch.autograd.Variable(torch.LongTensor(predict))
          cnn_outputs = cnn(input_data).cuda()
          torch.argmax()

    # print("length of Test[0]",len(Test[0]))
    # print("length of Test",len(Test))
    # print(len(ID))


if __name__=="__main__":
    #main()
    predict()