import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 定义res类
class Residual(nn.Module):
    # 根据use_1_1conv的不同生成不同的模型
    # 当为False时,生成的就是普通的模型h(x)=f(x)
    # 当为True时,生成的是包含h(x)=f(x)+x,因为x就是1*1的卷积,用来调整通道和分辨率
    def __init__(self, input_channels, num_channels
                 , use_1_1conv=False, strides=1):
        super(Residual, self).__init__()
        # 根据结构
        # 第一个卷积步幅为2,高宽减半,通道数翻倍
        # 第二个卷积为k3p1,大小不变
        self.conv1 = nn.Conv2d(input_channels, num_channels
                               , kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels
                               , kernel_size=3, padding=1)
        if use_1_1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels
                                   , kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# # 测试输出大小
# # blk为通道数不变,所以默认使用的padding=1和stride=1,kernel=3,所以图像大小不变
# blk = Residual(3, 3)
# # 加上了x,大小和channel同样也是不变的
# blk2 = Residual(3, 3, use_1_1conv=True)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# Y2 = blk(X)
# print(Y.shape, Y2.shape)
# # torch.Size([4, 3, 6, 6]) torch.Size([4, 3, 6, 6])
#
# # 如果我们想要将图像大小缩小一倍,那么strides需要=2,我们的channel就要从3变为6
# blk = Residual(3, 6, use_1_1conv=True, strides=2)
# print(blk(X).shape)

# ResNet的前两层同跟GoogLeNet中的一样,
# 第一层: output_channel=64,strides=3,kernel=7
# 第二层: output_channel=64,strides=1,kernel=3
# 不同之处在于ResNet每个卷积层之后都增加了BN, 当然x那一个identity mapping是不需要的
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
                   , nn.BatchNorm2d(64)
                   , nn.ReLU()
                   , nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# GoogLeNet后面接了4个Inception块组成的模块,ResNet则使用4个由残差块组成的模块
# 每个模块使用若干同样输出通道数的残差块, 第一个模块的通道数同输入通道数相同,也就是没有降采样
# 所以第一个模块的first_block=True,代表创建stage里的第一个block不需要降采样,也就是不需要x
# 由于之前已经使用了步幅为2的最大汇聚层,所以无需减小高和宽
# 之后的每个模块在第一个残差块里将上一个模块的第一个残差块的通道数翻倍,并将高和宽减半
# 可以看我来的ResNet-N结构图的stage
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels
                                , use_1_1conv=True, strides=2))
        else:
            blk.append(Residual(input_channels, num_channels))
        input_channels = num_channels
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 与GoogLeNet同样,在ResNet中加入全局平均汇聚层,以及全连接层输出
net = nn.Sequential(b1, b2, b3, b4, b5
                    , nn.AdaptiveAvgPool2d((1, 1))
                    , nn.Flatten()
                    , nn.Linear(512, 10))

X = torch.rand(size=(1, 1, 224, 224))
for i, layer in enumerate(net):
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', layer._modules.items(), X.shape)
