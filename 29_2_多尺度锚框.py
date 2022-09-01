# --coding:utf-8--
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

img = plt.imread('./dataset/catdog.jpg')
h, w = img.shape[:2]
print(h, w)


# 在特征图（fmap）上生成锚框（anchors）每个单位（像素）作为锚框的中心
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)


# display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# plt.show()

display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.show()
