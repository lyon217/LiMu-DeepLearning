import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from d2l import torch as d2l
import matplotlib.pyplot as plt

# @save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

data_dir = r'./dataset/banana-detection'


def read_data_bananas(is_train=True):
    # data_dir = d2l.download_extract('banana-detection')
    csv_name = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')

    csv_data = pd.read_csv(csv_name)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir,
                         'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含类别,左上角x,左上角y,右下角x,右下角y
        # 其中所有图像都具有相同的香蕉类(索引为0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananaDataset(Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""

    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read' + str(len(self.features)) + (f'training examples' if is_train
                                                  else f'validation examples'))

    def __getitem__(self, item):
        return self.features[item].float(), self.labels[item]

    def __len__(self):
        return len(self.features)


# 定义load_data_bananas函数,来为训练集和测试集返回两个数据加载器实例
# 对于测试集,无须按照随机顺序读取
def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananaDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananaDataset(is_train=False),
                                           batch_size, shuffle=False)
    return train_iter, val_iter


batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)


# torch.Size([32, 3, 256, 256]) torch.Size([32, 1, 5])

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
print(imgs.shape)
axes = show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

plt.show()
