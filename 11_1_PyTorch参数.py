import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8)
                    , nn.ReLU()
                    , nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))
# tensor([[-0.4867], [-0.5601]], grad_fn= < AddmmBackward0 >)

# 参数访问: 可以将每一层中的权重的值拿出
print(net[2].state_dict())  # net[2]指的其实就是nn.Sequential中的nn.Linear(8,1)
# OrderedDict([('weight', tensor([[-0.1817,  0.3414,  0.2506,  0.0706, -0.0223, -0.2458,  0.0457, -0.0273]]))
#            , ('bias', tensor([0.2315]))])

# 当然可以直接访问某一个具体的参数
print(type(net[2].bias))
# <class 'torch.nn.parameter.Parameter'>  # Parameter是一个可优化的参数
print(net[2].weight.data)
# tensor([[-0.2909,  0.2428,  0.3192,  0.2057, -0.2021,  0.2531,  0.0668, -0.0114]])
print(net[2].bias)
# Parameter containing:
# tensor([0.1798], requires_grad=True)
print(net[2].bias.data)
# tensor([0.1798])
print(net[2].weight.grad == None)
# True  # 因为还没有开始训练,所以暂时没有梯度

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
print(*[(name, param.shape) for name, param in net.named_parameters()])
# ('0.weight', torch.Size([8, 4]))
# ('0.bias', torch.Size([8]))
# ('2.weight', torch.Size([1, 8]))
# ('2.bias', torch.Size([1]))
# 其实可以看出: nn.Linear在初始化的时候weight和bias都是以Parameter类进行初始化的
# 但是nn.ReLU()是没有的
print(net.state_dict()['2.bias'].data)


# tensor([-0.2137])

# 从嵌套块中收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8)
                         , nn.ReLU()
                         , nn.Linear(8, 4)
                         , nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(3):
        net.add_module(f'block{i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
X = torch.rand(1, 4)
print(rgnet(X))  # tensor([[-0.2893]], grad_fn=<AddmmBackward0>)
print(rgnet)


# Sequential(
#   (0): Sequential(
#     (block0): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block1): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block2): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#   )
#   (1): Linear(in_features=4, out_features=1, bias=True)
# )


# 内置初始化 -> 怎么样初始我们的参数
# 1. 符合正态分布的初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


# apply函数的作用可以理解为: 给你一种方式(一个函数)让整个神经网络进行遍历修改
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])


# 2. 全部都是固定值
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])


# 3. 对某些块应用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:2])

# 别的方法:
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

# 参数绑定
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8)
                    , nn.ReLU()
                    , shared
                    , nn.ReLU()
                    , shared
                    , nn.ReLU()
                    , nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])