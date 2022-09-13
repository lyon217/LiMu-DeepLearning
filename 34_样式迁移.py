# --coding:utf-8--

import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt
from mytools import Animator

d2l.set_figsize()
content_img = Image.open('./dataset/rainier.jpg')
plt.imshow(content_img)
plt.show()

style_img = Image.open('./dataset/autumn-oak.jpg')
plt.imshow(style_img)
plt.show()

# 预处理和后处理
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)


def postprocess(img):
    img = img[0].to(rgb_std.device)
    # torchvision.transforms.functional_tensor.normalize的实现中的操作tensor.sub_(mean).div_(std)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


# 抽取图像特征
# VGG系列对于特征的抽取还是不错的，原论文中使用的就是VGG，也可能是因为VGG比较的规整
pertrained_net = torchvision.models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
# 样式层的匹配选用了多层，因为我们既想要匹配一些局部的细节，例如笔触动画油画，也想去匹配一些高级的图片的信息
# 内容层的匹配只选用了一层，是因为越往后的层富含了越高级的语义，我们也允许内容有一定的变形，所以没有选用比较低的层
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

net = nn.Sequential(*[pertrained_net.features[i]
                      for i in range(max(content_layers + style_layers) + 1)])


# 直接调用net(X)只会得到最后一层的结果，所以逐层计算
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


# get_content函数对内容图像抽取内容特征，get_styles函数对风格图像抽取风格特征
# 因为在训练时无需改变训练的VGG的参数，所以可以在训练开始之前就提取出内容特征和风格特征
# 由于合成图像是风格迁移所需迭代的模型参数，我们只能在训练过程中通过调用extract_features函数
# 来抽取合成图像的内容特征和风格特征
def get_contents_features(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles_features(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, style_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, style_Y


# 内容损失 -> 通过平方误差函数衡量合成图像与内容图像在内容特征上的差异
def content_loss(Y_hat, Y):
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量
    return torch.square(Y_hat - Y.detach()).mean()


def gram(X):
    # tensor.numel()返回tensor元素总数
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean()
                  +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


content_weight, style_weight, tv_weight = 1, 100000, 10


def compute_loss(X, contents_Y_hat, style_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [content_loss(Y_hat, Y) * content_weight
                  for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight
                for Y_hat, Y in zip(style_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 初始化合成图像
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__()
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# 训练模型
def train(X, content_Y, styles_Y, device, lr, num_epoches, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = Animator.Animator(xlabel='epoch', ylabel='loss',
                                 xlim=[10, num_epoches], ylim=[0, 20],
                                 legend=['content', 'style', 'TV'],
                                 ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epoches):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat,
                                                     content_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)),
                                     float(tv_l)])
    return X


device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents_features(image_shape, device)
_, styles_Y = get_styles_features(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.11, 1000, 50)
d2l.plt.show()
