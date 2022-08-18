import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l
from torch.utils import data

train_data = pd.read_csv('./dataset/kaggle_house_pred_train.csv')
test_data = pd.read_csv('./dataset/kaggle_house_pred_test.csv')

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 第一个feature是id,没有任何的预测信息,所以直接删除
# concat是直接将训练姐和测试机放到一起进行预处理,然后再分开
# 此处删除了train_data的第一列id和最后一列的label标签,test_data删除了第一列的id
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
#    Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
# 0   1          60       RL         65.0       WD        Normal     208500
# 1   2          20       RL         80.0       WD        Normal     181500
# 2   3          60       RL         68.0       WD        Normal     223500
# 3   4          70       RL         60.0       WD       Abnorml     140000
# 数据预处理 -> 标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 此处其实是计算的train_data和test_data的合并数据的每一列的期望和方差然后进行的标准化
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
# 进行标准化之后,均值就是0了,然后我们再对nan进行填0,其实就是填充均值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 独热编码替换,可以看到 最终生成了331个特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)  # (2919, 331)

# 从pandas中提取NumPy格式,并将其转换为张量
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values
                              , dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values
                             , dtype=torch.float32)
train_labesl = torch.tensor(train_data.SalePrice.values.reshape(-1, 1)
                            , dtype=torch.float32)
# print(train_features.shape, test_features.shape, train_labesl.shape)
# torch.Size([1460, 331]) torch.Size([1459, 331]) torch.Size([1460, 1])

loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    return nn.Sequential(nn.Linear(in_features, 1))


# 我们更关心相对误差,因为有些情况下,比如房价,一套房子可能预测100万,实际卖90万,有的房子可能预测10万,实际卖了9万,
# 所以他们的量纲是不太一致的,所以更关心相对误差 y* = (y-y^) / y,而不是绝对误差 y - y^
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, lr, weight_decay, batch_size):
    train_loss, test_loss = [], []

    train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_loss.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_loss.append(log_rmse(net, test_features, test_labels))

    return train_loss, test_loss


# K折交叉验证,有助于我们选择模型参数和调整
# 定义一个函数,在K折交叉验证过程中返回第i折的数据,也就是说它选择第i折作为验证数据,其余作为训练数据
# 当然,如果我们的数据集大得多,还会有别的解决方法
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # 每折样本的size
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        # i代表的是验证集,当j==i时,将该折放进X_valid中,当作验证集,其余的放进X_train,当作训练集
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# 训练K次,返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    # 分别是train_loss和valid_loss的总和
    # 两个总和 /k 后得到的就是每一轮的loss
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_loss, valid_loss = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_loss[-1]
        valid_l_sum += valid_loss[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1))
                     , [train_loss, valid_loss]
                     , xlabel='epoch'
                     , ylabel='rmse'
                     , xlim=[1, num_epochs]
                     , legend=['train', 'valid']
                     , yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_loss[-1]):f}, '
              f'验证log rmse{float(valid_loss[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labesl, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')
d2l.plt.show()