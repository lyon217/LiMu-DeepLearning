# --coding:utf-8--
import math
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from mytools.loaddataset import myLoadTimeMachine
from mytools import Timer, Accumulator, Animator

batch_size, num_steps = 32, 35
train_iter, vocab = myLoadTimeMachine.load_data_iter_time_machine(batch_size=batch_size,
                                                                  num_steps=num_steps)

# 独热编码
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
# 就是将0 和2 作为index在one_hot中进行编码
# 变为1, 0, 0....
# 和  0, 0, 1.....
# 我们每次采样的mini batch形状都是二维张量（批量大小，时间步数），one_hot将这样一个小批量转换成三维
# 最后一个维度等于词表大小，而且我们经常转换输入的维度，以便获得形状为(时间步数，批量大小，此表大小)的输出
# 这将使我们能够更方便地通过最外层的维度，一步一步地更新小批量数据的隐状态
# 个人理解：类比图像batch的shape(batch_size, channel, image),这里我们知道，为了能够并行处理，
# 我们加入了batch_size维度，使得很多个单位(channel,image)能够一起处理
# 但是对于sequence的batch的shape(num_steps, batch_size, one_hot)来讲，
# 每一个step都是有先后顺序的，不是并列的关系，所以我们将这一步的整个计算不是按照batch来分，
# 而是按照step来分，每一个step都是整个batch的一步计算，
# 每个batch拿出来一步来根据当前的onehot计算误差或者是结果(总共计算step次)
# 所以这也是为什么要把steps提前，让batch_size和one_hot放到一起，因为每一个(batch_size,one_hot)都是一步
X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape)  # torch.Size([5, 2, 28])


# 初始化模型参数
# 隐藏单元数num_hiddens是一个可调超参数
def get_params(vocab_size, num_hiddens, device):
    # 输入和输出都来自相同的词表，所以ont_hot之后，输出和输出都是28，都和vocab_size一致
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_h_out = normal((num_hiddens, num_outputs))
    b_out = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_h_out, b_out]
    for param in params:
        param.requires_grad_(True)
    return params


# 定义函数：在初始化时返回隐藏状态
# 因为在0时刻时，时没有隐藏状态的，所以要初始化一个，这样才能够一会传递下去,直接为0或者随机都是可以的
# 这里返回一个tuple，RNN的隐藏状态其实就是一个tensor，但是到后面的LSTM的话就不是了，为了统一，就返回tuple
def init_rnn_hidden_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device),


# rnn函数定义了一个时间步内计算隐状态和输出，
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_h_out, b_out = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_h_out) + b_out
        outputs.append(Y)
    # Y.shape应该就(batch_size,vocab_size),然后再cat之后变为(batch_size*num_step, vocab_size)
    return torch.cat(outputs, dim=0), (H,)


# 一个类来包装一下
class RNNModelScratch:
    """从0开始实现的RNN网络模型"""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 这里的X的shape就是批量大小乘以时间步数
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# 测试 class RNNMOdelScratch
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(),
                      get_params=get_params,
                      init_state=init_rnn_hidden_state,
                      forward_fn=rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)


# torch.Size([10, 28]) 1 torch.Size([2, 512])


# 预测
# prefix是一个用户提供的包含多个字符的字符串，在循环遍历prefix中的开始字符时，我们不断地将隐状态传递到下一个时间步
# 但是不生成任何输出，这被称为预热(warm-up)期，因为在此期间模型会自我更新(例如，更新隐状态)，但不会进行预测，
# 预测期结束后，隐状态的值通常比刚开始的初始值更适合预测，从而预测字符并输出他们
def predict_ch8(prefix, num_preds, net, vocab, device):
    # num_preds是往后预测的词元的个数
    # 初始化state
    state = net.begin_state(batch_size=1, device=device)
    # 取prefix的第一个词元作为最后一个输出,vocab[]对应的getitem中调用的是.token_to_idx函数
    outputs = [vocab[prefix[0]]]

    # get_input函数的功能就是取上一个output的值然后将其变为ndim=2的tensor
    def get_input():
        return torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 开始warm-up，每次取一个prefix词元
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # dim=1 -> 得到每一行的argmax
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print(predict_ch8('time traverller ', 10, net, vocab, d2l.try_gpu()))


# 进行梯度剪裁
def grad_clipping(net, theta):
    """梯度裁剪"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期"""
    state, timer = None, Timer.Timer()
    metric = Accumulator.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 训练过程使用API来实现
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = Animator.Animator(xlabel='epoch', ylabel='perplexity',
                                 legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda perfix: predict_ch8(perfix, 50, net, vocab, device)
    # 预测和训练
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度{ppl:.1f},{speed:.1f}词元/秒{str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
d2l.plt.show()

net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_hidden_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
d2l.plt.show()