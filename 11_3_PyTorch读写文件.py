import torch
from torch import nn
from torch.nn import functional as F

# 加载和保存张量
x = torch.arange(4)
torch.save(x, "x-file")

x2 = torch.load("x-file")
print(x2)

# 存储一个张量list, 然后读回内存
y = torch.zeros(4)
torch.save([x, y], 'x-file')
x2, y2 = torch.load('x-file')
print(x2, y2)

# 存储一个dict,然后读回
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)


# pytorch拥有一定的局限性,不方便把整个模型的定义存储下来,
# 别的解决方案也是有的, 例如torchscript
# 那如果我们使用pytorch进行存储,其实我们真正要存储的其实就是权重
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(2, 20)
Y = net(X)
# 将模型的参数存储为一个叫做"mlp.params"的文件
torch.save(net.state_dict(), 'mlp.params')

# 重新加载mlp.params
# 需要注意的是,net.state_dict()其实得到的就只有参数
# 所以我们如果要将存储的参数重新加载进来,还需要提前拿到网络模型的定义
clone = MLP()  # 这一步其实也会按照网络模型的定义进行初始化,但是我们不用管
clone.load_state_dict(torch.load("mlp.params"))
print(clone.eval())
# MLP(
#   (hidden): Linear(in_features=20, out_features=256, bias=True)
#   (output): Linear(in_features=256, out_features=10, bias=True)
# )

Y_clone = clone(X)
print(Y_clone == Y)