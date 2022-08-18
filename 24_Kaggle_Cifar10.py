import collections
import math
import os.path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch import nn
import random
import mytools
from mytools import resnet, Timer, Animator, Accumulator, Train

# #@save
# d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
#                                 '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
#
# # 如果你使用完整的Kaggle竞赛的数据集，设置demo为False
demo = True
#
# if demo:
#     data_dir = d2l.download_extract('cifar10_tiny')
# else:
#     data_dir = '../data/cifar-10/'

random.seed(10)

label_id = {'airplane': 1, 'automobile': 2, 'bird': 3, 'cat': 4, 'deer': 5,
            'dog': 6, 'frog': 7, 'horse': 8, 'ship': 9, 'truck': 10}


# 获得每一个str类型的类别对应的index
def get_label_id(label_name):
    return label_id.get(label_name, 0)


# 获得数据集对应的标签的dict,每一个元素对应的是-> id:label_id
def read_csv_train_label(filename,
                         dataset_name='kaggle_cifar10_tiny',
                         dataset_foldername='dataset'):
    with open(os.path.join(os.getcwd(),
                           dataset_foldername,
                           dataset_name, filename), 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return collections.OrderedDict((id, get_label_id(label_name)) for id, label_name in tokens)


ids_labels = read_csv_train_label("trainLabels.csv")
# print(ids_labels)
print('训练样本:', len(ids_labels))
print('leibls:', len(set(ids_labels.values())))


# 获得每一个种类labelid对应的n
def get_type_num():
    return collections.Counter(ids_labels.values())


# 组织数据  validation(验证集)
# valid_ratio是训练样本和验证样本的占比,
# 例如0.1就是从500张原始的训练集中随机挑选50张作为validation然后450张作为真正的训练集
# n是样本中最少的类别中的图像数量
def copyfile(filename, target_dir):
    # 将文件复制到目标目录:
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


idx = None
counts = None
ids = None
labels = None


def org_id_fromdict2list():
    global counts, ids, labels
    # ids: 所有的id
    ids = list(ids_labels.keys())
    # labels: 所有的id对应的label
    labels = list(ids_labels.values())
    counts = collections.defaultdict(lambda: [])
    for i, label in enumerate(labels):
        counts[label].append(ids[i])


# 组织数据后,将validation放到原始数据集文件夹中的valid文件夹中
# 最终的原始文件夹中存在的文件应为: train test valid
def reorg_train_valid1(data_dir, valid_ratio):
    global counts, idx
    min_n = min(get_type_num().values())
    # print(min_n) # 85
    n_valid_per_label = max(1, math.floor(min_n * valid_ratio))
    label_count = {}
    # 从0-84中选择n_valid_per_label个作为选择的每个类被的下标
    idx = random.sample(range(min_n), n_valid_per_label)

    for item in counts.items():
        for i in idx:
            '''copyfile(os.path.join(data_dir, 'train', str(item[1][i]) + '.png')
                     .replace('\\', '/'),
                     os.path.join(data_dir, 'valid', str(item[0]))
                     .replace('\\', '/'))'''
    return n_valid_per_label


def reorg_train_valid2(data_dir):
    global counts, idx
    valid_list = []
    for item in counts.items():
        for i in idx:
            valid_list.append(item[1][i])

    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        current_id = int(train_file.split('.')[0])
        if train_file.endswith('.png') and (current_id not in valid_list):
            copyfile(os.path.join(data_dir, 'train', train_file),
                     os.path.join(data_dir, 'train', str(labels[current_id - 1])))


batch_size = 32 if demo else 128
valid_ratio = 0.1
data_dir = './dataset/kaggle_cifar10_tiny'

org_id_fromdict2list()
reorg_train_valid1(data_dir, valid_ratio)
reorg_train_valid2(data_dir)

# 图像增广,原始图像为32*32
transform_train = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])])

# 测试集我们只做标准化,以消除评估结果中的随机性
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 读取数据集
train_ds, valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, folder).replace('\\', '/'),
    transform=transform_train) for folder in ['train', 'valid']]

test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'test').replace('\\', '/'), transform=transform_test)]

train_iter = torch.utils.data.DataLoader(train_ds, batch_size,
                                         shuffle=True, drop_last=True)
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size,
                                         shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size,
                                        shuffle=False, drop_last=True)


def get_net():
    num_class = 10
    net = resnet.resnet18(num_class=num_class, in_channels=3)
    return net


loss = nn.CrossEntropyLoss(reduction='none')


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = Animator.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                 legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = Train.train_batch_ch13(net, features, labels,
                                            loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = Train.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                     f' examples/sec on {str(devices)}')


devices, num_epochs, lr, wd = mytools.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

net, preds = get_net(), []

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: valid_ds.classes[x])
df.to_csv('submission.csv', index=False)

plt.show()
