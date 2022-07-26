import torch
from torch import nn
from d2l import torch as d2l


# 计算二维互相关 X为原始图像, K为kernel卷积核大小
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))


X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X, K)
print(Y)

# 我们已经有了最原始的X和X通过(1, -1)卷积之后的结果Y
# 然后我们将Y作为输出,去重新使用nn.Conv2d()构造一个卷积层
# 然后训练,会发现最终的结果conv2d.weight.data,也就是我们自己训练出来的结果
# 与我们最原始的(1, -1)卷积核的数值是差不多的
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch{i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))


# 填充和步幅
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    print('Y.shape:', Y.shape)
    # Y.shape: torch.Size([1, 1, 8, 8])
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)
# torch.Size([8, 8])

# 填充不同高度和宽度
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)


# torch.Size([8, 8])


# 会出现重复学习的情况,所以有了深度可分离卷积核


# 多输入多输出通道
def corr2d_mul_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))


# X-> (2, 3, 3) 2通道,每通道 3*3
X = torch.tensor([[[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]])
# K->kernel ->(2, 2, 3) 2通道, 每通道2*2
K = torch.tensor([[[0.0, 1.0],
                   [2.0, 3.0]],
                  [[1.0, 2.0],
                   [3.0, 4.0]]])

print(corr2d_mul_in(X, K))


# tensor([[ 56.,  72.],
#         [104., 120.]])


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y