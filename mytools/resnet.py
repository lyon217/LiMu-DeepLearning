import d2l.torch
import torch
from torch import nn
from torch.nn import functional as F


# 该类为一个stage中的一个残差块
# 每一个stage中的残差块的个数是不一样的
# 通常情况下,第一个stage的残差块的channels都是64
class Residual(nn.Module):
    def __init__(self, input_channels, out_channels, use_1_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)

        # 通常情况下,每一个stage的第一个残差块的是需要经过1*1步长为2的卷积层的降采样的,
        # 如果需要1_1conv,则需要in_cha,out_cha,k1s1的卷积的
        # 这个降采样的作用主要是为了使channel变大,而不是传递特征映射的原始的X的
        if use_1_1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels,
                                   kernel_size=1, stride=strides)
        # 如果不需要,则conv3=None,不做任何操作,
        # 然后当需要传递原始特征映射X时,直接将推理得到的结果+X即可,在forward中显示
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 与24对应 : 需要传递原始特征映射X时,直接将推理得到的结果+X即可,在forward中显示
        if self.conv3:
            X = self.conv3(X)
        Y += X
        # 需要传递原始特征映射X时,要在relu前+X
        return F.relu(Y)
        # 整体还是可以看出每一个残差块的过程为:
        # conv1->bn1->relu->conv2->bn2->(+X)->relu


def resnet18(num_class, in_channels=1):

    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            # 每一个stage中,第一个residual的channel都需要扩大一倍
            # ,但是第一个stage不需要,因为channel都是64
            # 所以除了第一个stage外,其他stage的residual都需要use_1_1conv
            # stride=2是因为前后两个stage的channel要扩大所以这里是stride=2,channel变大一倍
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1_1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
    net.add_module('resnet_block2', resnet_block(64, 128, 2))
    net.add_module('resnet_block3', resnet_block(128, 256, 2))
    net.add_module('resnet_block4', resnet_block(256, 512, 2))
    net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module('fc', nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_class)))
    return net