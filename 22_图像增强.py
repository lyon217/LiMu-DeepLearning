import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from PIL import Image

image = Image.open('lena.jpg')
# print(np.array(image).shape)  # (2318, 1084, 3)

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip()
        , transforms.ToTensor()
     # scale是截图的原始图片的比例大小,ratio是截取图片大小的宽高比例
        , transforms.RandomResizedCrop((700, 700), scale=(0.3, 1))
     # brightness亮度0,5代表增加或减少,contrast对比度,saturation饱和度,hue色调
        , transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)])


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def apply(image, trans, num_rows=2, num_cols=5, scale=1.5):
    Y = [trans(image) for _ in range(num_rows * num_cols)]
    imshow(torchvision.utils.make_grid(Y, nrow=5))


apply(image, transform)