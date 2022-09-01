# --coding:utf-8--

import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt

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
