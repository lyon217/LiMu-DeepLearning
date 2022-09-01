import torch
from torch import nn
import mytools
from mytools.loaddataset import myLoadDataSetFashionMnist
from mytools import Train
from d2l import torch as d2l


# 构建块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 根据conv_acrh构建架构
# conv_arch有5个元素,代表5个VGG块, 例如第一个(1,64)代表该块只有1层,输出通道数为64
# 因为是maxpooling会将图像大小减半,224/2/2/2/2/2=7,所以conv_arch中最多只能有5个元素
# 下面是VGG11架构
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def VGG(conv_arch):
    conv_blck = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blck.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blck
                         , nn.Flatten()
                         , nn.Linear(out_channels * 7 * 7, 4096)
                         , nn.ReLU()
                         , nn.Dropout(0.5)

                         , nn.Linear(4096, 4096)
                         , nn.ReLU()
                         , nn.Dropout(0.5)

                         , nn.Linear(4096, 10))


net = VGG(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape :\t', X.shape)

# 由于VGG-11比AlexNet计算量更大,所以我们构建一个通道数较少的网络
# raito=4就是所有的通道数都除以4
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = VGG(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = myLoadDataSetFashionMnist.my_load_data_fashion_mnist(batch_size, resize=224)

Train.train_ch6(net, train_iter, test_iter, num_epochs, lr, mytools.try_gpu())
d2l.plt.show()
