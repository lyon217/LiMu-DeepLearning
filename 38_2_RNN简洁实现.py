# --coding:utf-8--
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from mytools.loaddataset import myLoadTimeMachine

batch_size, num_steps = 32, 25
train_iter, vocab = myLoadTimeMachine.load_data_iter_time_machine(batch_size, num_steps)

# 构造一个具有256隐藏单元的单隐藏层的循环神经网络rnn_layer
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 使用张量来初始化隐藏状态，shape是（隐藏层数，批量大小，隐藏单元数）
state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)  # torch.Size([1, 32, 256])
# 通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。
# 需要强调的是，rnn_layer的“输出”（Y）不涉及输出层的计算： 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)

# .....未完成
