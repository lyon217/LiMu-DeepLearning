import os
import cv2
import time
import math

import timm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim import Adam, AdamW
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
import urllib
import pickle
import torch.nn.functional as F
import seaborn as sns
import random
import sys
import gc
import shutil
from tqdm.autonotebook import tqdm
import albumentations as A
from albumentations import pytorch as AT

import scipy.special

sigmoid = lambda x: scipy.special.expit(x)
from scipy.special import softmax

import torch_utils as tu

import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------
SEED = 42
base_dir = '../input/'
# torch.backends.cudnn.benchmark👇
# https://cloud.tencent.com/developer/article/1668398
tu.tools.seed_everything(SEED, deterministic=False)
# torch.backends.cudnn.deterministic是啥？顾名思义，
# 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
# 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的，

# choice gpu id if you have more than one GPU
tu.tools.set_gpus('0')  # choice gpu id if you have more than one GPU
# -------------------------------------------------------------------------------------------

# 自动备份实验,保证数据不丢失
EXP = 1
while os.path.exists('../exp/exp%d' % EXP):
    EXP += 1
os.makedirs('../exp/exp%d' % EXP)

# -------------------------------------------------------------------------------------------

# Param 可调参数
CLASSES = 176
FOLD = 5
BATCH_SIZE = 64
ACCUMULATE = 1
LR = 3e-4
EPOCH = 36
DECAY_SCALE = 20.0
MIXUP = 0  # 0 to 1

# -------------------------------------------------------------------------------------------

# Dataset
# 定义数据集以及增广,使用了RandAugment
train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=1, p=0.5),
    tu.randAugment(),
    A.Normalize(),
    AT.ToTensorV2(),
])
test_transform = A.Compose([
    A.Normalize(),
    AT.ToTensorV2(),
])


# -------------------------------------------------------------------------------------------

class LeavesDataset(Dataset):
    def __init__(self, df, label_encoder, data_path='../input', transform=train_transform):
        self.df = df
        self.data_path = data_path
        self.transform = transform
        self.df.label = self.df.label.apply(lambda x: label_encoder[x])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path, label = self.df.image[idx], self.df.label[idx]
        img_path = os.path.join(self.data_path, img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)
        img = self.transform(image=img)['image']
        return img, label


# -------------------------------------------------------------------------------------------
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
train_df.head()

# K Fold Split Train-Val dataset
skf = StratifiedKFold(n_splits=FOLD, random_state=SEED, shuffle=True)
train_folds = []
valid_folds = []
for train_idx, valid_idx in skf.split(train_df.image, train_df.label):
    train_folds.append(train_idx)
    valid_folds.append(valid_idx)
    print(len(train_idx), len(valid_idx))

# Training Loop
# 混合精度训练,节约显存获得更大的batch size,并节约训练时间,损失较小,甚至能因为大batch而提升性能
from torch.optim.lr_scheduler import CosineAnnealingLR

scaler = torch.cuda.amp.GradScaler()  # for AMP training

log = open('../exp/exp' + str(EXP) + '/log.txt', 'w')
log.write('SEED%d\n' % SEED)
cv_losses = []
cv_metrics = []


def train(fold):
    pass


for fold in range(FOLD):
    print('\n*********** FLOD %d ***********\n' % fold)
    labels = train_df.label.unique()
    label_encoder = {}
    for idx, name in enumerate(labels):
        label_encoder.update({name: idx})

    trainset = LeavesDataset(train_df.iloc[train_folds[fold]].reset_index(),
                             label_encoder, base_dir, train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=2,
                                               shuffle=True, drop_last=True,
                                               worker_init_fn=tu.tools.worker_init_fn)
    valset = LeavesDataset(train_df.iloc[valid_folds[fold]].reset_index(),
                           label_encoder, base_dir, test_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    ################### Model #####################
    model_conv = tu.ImageModel(name='tf_efficientnetv2_l_in21ft1k', pretrained=True,
                               num_feature=2048, classes=CLASSES)
    model_conv.cuda()
    model_conv = torch.nn.DataParallel(model_conv)

    ################### Optim #####################
    optimizer = tu.RangerLars(model_conv.parameters(), lr=LR, weight_decay=2e-4)

    if MIXUP:
        criterion = tu.SoftTargetCrossEntropy()
    else:
        criterion = tu.LabelSmoothingCrossEntropy()

    criterion_test = nn.CrossEntropyLoss()

    T = len(train_loader)  # Accumulate * Epoch # cycle

    scheduler = CosineAnnealingLR(optimizer, T_max=T, eta_min=LR / DECAY_SCALE)
    val_loss, val_acc = train(fold)
    cv_losses.append(val_loss)
    cv_metrics.append(val_acc)
    torch.cuda.empty_cache()

cv_loss = sum(cv_losses) / FOLD
cv_acc = sum(cv_metrics) / FOLD
print('CV loss:%.6f  CV precision:%.6f' % (cv_loss, cv_acc))
log.write('CV loss:%.6f  CV precision:%.6f\n\n' % (cv_loss, cv_acc))

log.close()
tu.tools.backup_folder('.''../exp/exp%d/src' % EXP)


def train_model(epoch, verbose=False):
    model_conv.train()
    avg_loss = 0.
    optimizer.zero_grad()
    if verbose:
        bar = tqdm(total=len(train_loader))
    mixup_fn = tu.Mixup(prob=MIXUP, switch_prob=0.0, onehot=True, label_smoothing=0.05, num_classes=CLASSES)
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.float().cuda(), labels.cuda()
        if MIXUP:
            imgs_train, labels_train = mixup_fn(imgs_train, labels_train)
        with torch.cuda.amp.autocast():
            output_train, _ = model_conv(imgs_train)
            loss = criterion(output_train, labels_train)
        scaler.scale(loss).backward()
        if ((idx + 1) % ACCUMULATE == 0):  # Gradient Accumulate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        avg_loss += loss.item() / len(train_loader)
        if verbose:
            bar.update(1)
    if verbose:
        bar.close()
    return avg_loss


def tes_model():
    avg_val_loss = 0.
    model_conv.eval()
    y_true_val = np.zeros(len(valset))
    y_pred_val = np.zeros((len(valset), CLASSES))
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            imgs_vaild, labels_vaild = imgs.float().cuda(), labels.cuda()
            output_test, _ = model_conv(imgs_vaild)
            avg_val_loss += (criterion_test(output_test, labels_vaild).item() / len(val_loader))
            a = labels_vaild.detach().cpu().numpy().astype(np.int)
            b = softmax(output_test.detach().cpu().numpy(), axis=1)

            y_true_val[idx * BATCH_SIZE:idx * BATCH_SIZE + b.shape[0]] = a
            y_pred_val[idx * BATCH_SIZE:idx * BATCH_SIZE + b.shape[0]] = b

    metric_val = sum(np.argmax(y_pred_val, axis=1) == y_true_val) / len(y_true_val)
    return avg_val_loss, metric_val


def train(fold):
    best_avg_loss = 100.0
    best_acc = 0.0
    avg_val_loss, avg_val_acc = tes_model()
    print('pretrain val loss %.4f precision %.4f' % (avg_val_loss, avg_val_acc))

    ### training
    for epoch in range(EPOCH):
        print('lr:', optimizer.param_groups[0]['lr'])
        np.random.seed(SEED + EPOCH * 999)
        start_time = time.time()
        avg_loss = train_model(epoch)
        avg_val_loss, avg_val_acc = tes_model()
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t train_loss={:.4f} \t val_loss={:.4f} \t val_precision={:.4f} \t time={:.2f}s'.format(
            epoch + 1, EPOCH, avg_loss, avg_val_loss, avg_val_acc, elapsed_time))

        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model_conv.module.state_dict(), '../exp/exp' + str(EXP) + '/model-best' + str(fold) + '.pth')
            print('model saved!')

        print('=================================')
    timm.create_model()
    print('best loss:', best_avg_loss)
    print('best precision:', best_acc)
    return best_avg_loss, best_acc


class LeavesData_inference(Dataset):
    def __init__(self, df, data_path='mykaggle/classify-leaves/dataset/images',
                 transform=test_transform):
        self.df = df
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df.image[idx]
        img_path = os.path.join(self.data_path, img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img


test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
testset = LeavesData_inference(test_df, base_dir, tta_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)

TTA = 16
pred_list = []
for tta in range(TTA):
    for fold in range(FOLD):
        y_pred_val = np.zeros(len(testset), CLASSES)
        model_conv = tu.ImageModel(
            # name='tf_efficientnetv2_l_in21ft1k',
            pretrained=True,
            # num_feature=2048,
            classes=CLASSES)
        model_conv.cuda()
        model_conv.load_state_dict(torch.load(''))
        with torch.no_grad():
            for idx, imgs in enumerate(test_loader):
                imgs_test = imgs.float().cuda()
                output_test, _ = model_conv(imgs_test)
                b = softmax(output_test.detach().cpu().numpy(), axis=1)
                y_pred_val[idx * BATCH_SIZE:idx * BATCH_SIZE + b.shape[0]] = b
        pred_list.append(y_pred_val)
        print('TTA %d: Flod %d' % (tta, fold))

pred = sum(pred_list) / len(pred_list)
np.save()
label_decoder = {}
for k, v in label_encoder.items():
    label_decoder[v] = k

pred_label = np.argmax(pred, axis=1)
test_df['label'] = pred_label
test_df['label'] = test_df['label'].apply(lambda x: label_decoder[x])

test_df.to_csv('exp%d.csv'%EXP, index=False)

