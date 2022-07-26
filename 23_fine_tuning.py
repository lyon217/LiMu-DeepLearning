import os, torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import mytools
from mytools import Train
from d2l import torch as d2l


# 下载d2l的hotdog数据集
# d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
# data_dir = d2l.download_extract('hotdog')

# 用于展示make_grid之后的tensor
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class trans_square:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        image = args[0]
        C, H, W = image.shape
        pad_padh, pad_padw = 0, 0
        if (224 - H) % 2 != 0:
            pad_padh = 1
        if (224 - W) % 2 != 0:
            pad_padw = 1
        pad_H = int(abs(224 - H) // 2)
        pad_W = int(abs(224 - W) // 2)
        # 在0位置上再加一个轴,因为nn.ZeroPad2d的输入是(N, C, H_in, W_in)or(C, H_in, W_in)
        # image = image.unsqueeze(0)
        if H < 224:
            image = nn.ConstantPad2d((0, 0, pad_H, pad_H + pad_padh), 1)(image)
        if W < 244:
            image = nn.ConstantPad2d((pad_W, pad_W + pad_padw, 0, 0), 1)(image)
        # img = img.squeeze(0)
        # img = transforms.ToTensor()(img)
        # print(type(img))
        return image


class Crop2_300:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        img = args[0]
        # print(img.shape)
        C, W, H = img.shape[0], img.shape[1], img.shape[2]
        if W > 224 or H > 224:
            img = torchvision.transforms.Resize((224, 224))(img)
            print('-', img.shape)
            return img
        print('-', img.shape)
        return img


train_augs = transforms.Compose([transforms.RandomResizedCrop(224)
                                    , transforms.RandomHorizontalFlip()
                                    , transforms.ToTensor()
                                    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_augs = transforms.Compose([transforms.Resize(256)
                                   , transforms.CenterCrop(224)
                                   , transforms.ToTensor()
                                   , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_imgs = torchvision.datasets.ImageFolder('./dataset/hotdog/train/')
# train_imgs是一个list,list里面有两个对象,分别是训练集中对应的两个类,一个是热狗类,一个是非热狗类,每个类下就是一个图片
# 读进来的时候就是一个PIL, test_imgs同理
test_imgs = torchvision.datasets.ImageFolder('./dataset/hotdog/test/')

transf_for_display = transforms.Compose([transforms.ToTensor()
                                            , trans_square()
                                            , Crop2_300()])

hotdogs = [transf_for_display(train_imgs[i][0]) for i in range(8)]

print(type(hotdogs[0]))  # # <class 'torch.Tensor'>

# imshow(torchvision.utils.make_grid(hotdogs, nrow=4))

# 我们使用ImageNet数据集上与训练的ResNet-18作为源模型,在这里指定pretrained=True就会自动下载预训练的模型参数
# pretrained_net = torchvision.models.resnet18(pretrained=True)
pretrained_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
print(pretrained_net)

# 预训练的源模型实例包含许多特征层和一个输出层fc,此划分的主要目的是促进对输出层以外所有层的模型参数进行微调
# 下面给出了源模型的成员变量fc full connection 全连接层
# print(pretrained_net.fc)

# nn.Linear(in_features=512, out_features=1000, bias=True)
# 在ResNet的Global Average Pooling Layer 后,全连接层转换为1000个类别输出,之后,我们构建我们自己的的网络
# 它的定义方式与预训练源模型的定义方式相同,只是最终的输出层中的输出数量被设置为了我们的目标数据集中的类别数
# 使用pretrained_net来初始化我们自己的模型finetune_net的参数
finetune_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)  # 2就是out_features,指的就是hotdog的2类
# 可以看出,最后的fc我们是重新初始化的,是从头开始训练的,所以通常需要更高的学习率
# 一般我们可以设置该层的学习率lr是已经经过预训练的层的学习率的10倍
nn.init.xavier_uniform_(finetune_net.fc.weight)


# 微调模型
# 我们定义了一个train_fine_tuning,该函数使用微调,因此可以使用多次
def train_fine_tuning(net, lr, num_epochs=5, batch_size=128, param_group=True):
    train_iter = DataLoader(torchvision.datasets.ImageFolder('./dataset/hotdog/train/', transform=train_augs)
                            , batch_size=batch_size
                            , shuffle=True)
    test_iter = DataLoader(torchvision.datasets.ImageFolder('./dataset/hotdog/test/', transform=test_augs)
                           , batch_size=batch_size)
    devices = mytools.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        # 除了fc.weight和fc.bias以外的参数的学习率是1倍
        params_1x_lr = [param for name, param in net.named_parameters()
                        if name not in ['fc.weight', 'fc.bias']]
        # 除了params_1x_lr中的其他的参数都是10倍的lr
        trainer = torch.optim.SGD([{'params': params_1x_lr},
                                   {'params': net.fc.parameters(), 'lr': lr * 10}]
                                  , lr=lr, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)
        Train.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 我们使用较小的学习率,通过fine-tuning训练获得的模型参数
train_fine_tuning(finetune_net, 5e-5)

# 为了进行比较,我们定义了一个相同的模型,但是将其所有的模型参数初始化为随机值,由于整个模型需要从头进行训练,
# 因此我们使用更大的学习率
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
