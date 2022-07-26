import torch
import torch.nn.functional as F
from torch import nn


# 1. 自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
# tensor([-2., -1.,  0.,  1.,  2.])


# 2. 将自定义层作为组件构建到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))  #
print(Y.mean())


# 3. 带参数的层
class MyLinear(nn.Module):
    def __init__(self, n_input, n_output):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(n_input, n_output))
        self.bias = nn.Parameter(torch.zeros(n_output))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


dense = MyLinear(5, 3)
print(dense.weight)
# 使用自定义层直接执行正向传播计算
dense(torch.rand(2, 5))
# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))