import torch
import evaluate
class EncoderRNNWithVector(torch.nn.Module):
    def __init__(self, hidden_size, out_size, n_layers=1, batch_size=20):
        super(EncoderRNNWithVector, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了 BATCH FIRST
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

       # 加了一个线性层，全连接
        self.out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, word_inputs, hidden):
        # -1 是在其他确定的情况下，PyTorch 能够自动推断出来，view 函数就是在数据不变的情况下重新整理数据维度
        # batch, time_seq, input
        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)

        #print("inputs shape",inputs.shape) #([20, 10, 200])
        #print("hidden shape",hidden.shape) #hidden shape torch.Size([1, 20, 200])

        # hidden 就是上下文输出，output 就是 RNN 输出
        output, hidden = self.gru(inputs, hidden)

        output = self.out(output)

        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:, -1, :]

        return output, hidden

    def init_hidden(self):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(torch.zeros(self.n_layers,self.batch_size ,self.hidden_size))
        return hidden

