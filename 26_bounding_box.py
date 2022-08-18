import torch
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from d2l import torch as d2l


def set_figsize(figsize=(3.5, 2.5)):
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
catdog_img = plt.imread('./dataset/catdog.jpg')


# plt.imshow(catdog_img)
# plt.show()


# 定义两种表示方法之间进行转换的函数: box_corner_to_center从两角表示法转换为中心宽度表示法
def box_corner_to_center(boxes):
    """从(左上,右下)转换到(中间,宽度,高度)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 此时认为cx,cy,w,h的dim都是1,shape都是n
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # 经过stack(axis=-1)将原始的维度变为行,然后增加一个列维度,堆叠后变为n*4
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


# box_center_to_corner表示从中心表示法转换为两角表示法
def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - (w / 2)
    x2 = cx + (w / 2)
    y1 = cy - (h / 2)
    y2 = cy + (h / 2)
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]


# boxes = torch.tensor((dog_bbox, cat_bbox))
# print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

def bbox_to_rectangle(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1])
                             , width=bbox[2] - bbox[0]
                             , height=bbox[3] - bbox[1]
                             , fill=False
                             , edgecolor=color
                             , linewidth=2)


fig = d2l.plt.imshow(catdog_img)
fig.axes.add_patch(bbox_to_rectangle(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rectangle(cat_bbox, 'red'))
plt.show()
