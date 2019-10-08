import jieba
import torchtext.data as data
from config import args
import torch
from model.CNN import TEXTCNN
from evaluate import evaluate_classify,predict_all

def tokenizer(x):
    res=[w for w in jieba.cut(x)]
    return res

def get_dataset():
   path="data/"
   Text=data.Field(sequential=True,tokenize=tokenizer,fix_length=500,batch_first=True)
   Label=data.Field(sequential=False,use_vocab=False,batch_first=True)
   train,valid=data.TabularDataset.splits(path=path,train='train_after.csv',validation='dev_after.csv', format='csv',
                                          skip_header=True,
                                         fields=[('id',None),('title',None),('content',Text),('label',Label)])
   Text.build_vocab(train)
   print(Text.vocab.itos[0]) #<unk>
   print(Text.vocab.itos[1]) #<pad>
   print(Text.vocab.itos[100])
   args.vocab_size=len(Text.vocab)
   cnn = TEXTCNN(args).cuda()

   train_iter, dev_iter = data.BucketIterator.splits((train, valid), batch_sizes=(64, 64), sort_key=lambda x: len(x.content),
                                                sort_within_batch=False, repeat=False)
   criterion = torch.nn.CrossEntropyLoss()
   opt_Adam = torch.optim.Adam(cnn.parameters(), lr=args.lr, betas=(0.9, 0.99))
   for epoch in range(1, args.epoches):
     for step,batch in enumerate(train_iter):
      trainx = batch.content
      trainy = batch.label
      input_data = torch.autograd.Variable(torch.LongTensor(trainx)).cuda()
      output_labels = torch.autograd.Variable(torch.LongTensor(trainy)).cuda()
      cnn_outputs = cnn(input_data)
      outputs=torch.argmax(cnn_outputs,dim=-1)
      evaluate_classify(outputs,trainy)
      opt_Adam.zero_grad()
      loss = criterion(cnn_outputs, output_labels)
      loss.backward()
      opt_Adam.step()
      if step%20==0:
          predict_all(dev_iter,cnn)


if __name__ =="__main__":
     get_dataset()