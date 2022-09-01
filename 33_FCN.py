# --coding:utf-8--
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from mytools import Train
from mytools.loaddataset import myLoadDataSetVoc

# 我们使用ImageNet上pretrained的ResNet-18
# 但是最后基层包括全局平均汇聚层和全连接层是不需要的
pretrained_net = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
print(list(pretrained_net.children())[-3:])

# 我们创建一个全卷机网络net，赋值ResNet-18中大部分的预训练层，除了最后的全局平均汇聚和全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])
# 给定高度为320和宽度为480的输入，net的前向传播将输入的高和宽减小至原来的1/32，即10和15
X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)  # torch.Size([1, 512, 10, 15]) 可以看到宽高都缩小了32倍
# 接下来我们使用1×1卷积层将输出通道数转换为Pascal VOC2012数据集的21类
# 最后，我们需要将特征图的高度和宽度增加32倍，从而将其变回输入图像的高宽
# 由转置卷积尺寸公式 ns+k-2p-s = n'
# 所以此时，如果p=s/2(能整除) 并且 k=2s上述公式就变换为：ns+2s-s-s = ns = n'
# 这时会发现转置卷积之后的大小变为了原来的s倍，
# 又由于原始图像的大小经过resnet18的backbone之后变为了原来的1/32，
# 所以此时我们可以让转置卷积的s变为32，这样经过转置卷积之后的图像就可以变为原来的32倍
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))


# 双线性插值初始化转置卷积层 bilinear interpolation
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2  # 相当于除2然后向上取整
    if kernel_size % 2 == 1:  # 如果kernel_size为奇数
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
# 读取图像X，将上采样结果记作Y，为了打印图像，需要调整通道维度位置
img = torchvision.transforms.ToTensor()(d2l.Image.open('./dataset/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
# 可以看到转置卷积将图像的高宽分别放大了2倍
d2l.set_figsize((5, 5))
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
d2l.plt.show()
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)
d2l.plt.show()

# 全卷积网络中，我们用双线性插值的上采样初始化转置卷积层，对于1×1卷积层，我们使用Xavier初始化参数
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 读取数据集
batch_size, crop_size = 16, (320, 480)
train_iter, test_iter = myLoadDataSetVoc.load_data_voc(batch_size, crop_size)


# 训练
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


# try_all_gpus返回的是所有的GPU的list，如果没有gpu则返回cpu
num_epochs, lr, wd, devices = 20, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
Train.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
torch.save(net.module.state_dict(), './dataset/pthfile/FCN_VOC.pth')
d2l.plt.show()


# 预测
# 需要将输入图像在各个通道做标准化，并转成卷积神经网络所需要的四维输入格式
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    # 这里取dim=1是取的通道维度，
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


# 为了可视化预测的类别给每个像素，我们将预测类别映射回它们在数据集中的标注颜色
def label2image(pred):
    colormap = torch.tensor(myLoadDataSetVoc.VOC_COLORMAP, devices=devices[0])
    X = pred.long()
    return colormap[X, :]


test_image, test_labels = myLoadDataSetVoc.read_voc_images(is_train=False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_image[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permyte(1, 2, 0),
             pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1, 2, 0)]

d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
d2l.plt.show()
