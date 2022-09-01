# --coding:utf-8--
import torch
from torch import nn
from d2l import torch as d2l


def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j:j + w] += X[i, j] * K
    return Y


X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))

# 档X和K都是4维张量时，我们可以使用高级API获得相同的结果
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 填充，步幅和多通道

# 于常规卷积不同，在转置卷积中，填充被应用于的输出（常规卷积将填充应用于输入）
# 将将高宽两侧的填充数指定为1时，转置卷积的输出中将删除第1和最后的行和列
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
# tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)
print(tconv(X))

# 在转置卷积中，步幅被指定为中间结果（输出），而不是输入
# 也就是说，每计算一次，位置会往后或下位移stride个单位，导致最终的输出的尺寸会变大
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
# tensor([[[[0., 0., 0., 1.],
#           [0., 0., 2., 3.],
#           [0., 2., 0., 3.],
#           [4., 6., 6., 9.]]]], grad_fn=<ConvolutionBackward0>)
print(tconv(X))

# 对于多个输入和输出通道，转置卷积与常规卷积以相同的方式运作
# 假设input有C个通道，每个输入通道分配了一个kh*kw的卷积核
# 所以每一个输出通道都会对应C*kh*kw的卷积核（与普通卷积完全相同）
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)  # True

# ------------------------------------------------------------------------------------
# 与矩阵变换的关系 -> 转置卷积为何以转置命名？
# 我们首先定义一个3*3的输出X,和一个2*2的kernel K,然后使用corr2d函数计算卷积输出Y
X = torch.arange(9).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)  # 计算普通卷积
# tensor([[27., 37.],
#         [57., 67.]])
print(Y)


# 然后我们将卷积核K重写为包含大量0的稀疏权重矩阵W，权重矩阵的形状是(4,9),其中非0元素来自卷积核K
def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W


W = kernel2matrix(K)  # 可以理解为手动构造一个W
# tensor([[1., 2., 0., 3., 4., 0., 0., 0., 0.],
#         [0., 1., 2., 0., 3., 4., 0., 0., 0.],
#         [0., 0., 0., 1., 2., 0., 3., 4., 0.],
#         [0., 0., 0., 0., 1., 2., 0., 3., 4.]])     4×9
print(W)
# 将W转置为 9×4
Z = trans_conv(Y, K)
# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True]])
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))
# 理解：其实普通卷积可以理解为一开始的  X   @   K  =>  Y
#                                3*3     2*2    2*2
# 将X K Y都展开为一维向量的形式      1*9     9*4    1*4
#                                         ↓（K的1*4卷积要进行9次，所以是9*4）
# 那转置卷积可以理解为从Y求X的一个过程：
# 从 X  @  K  =>  Y  变为： Y   tran_conv   W   =>   X  那此时的W就是K.T
# ：                       Y   tran_conv   W   =>   X
#                         1*4             4*9      1*9
#                                          ↓就是K.T
