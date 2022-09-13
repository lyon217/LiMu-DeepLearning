# --coding:utf-8--
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch.utils import data

T = 1000
time = torch.arange(start=1, end=T + 1, step=1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(mean=0, std=0.2, size=(T,))
# plt.figure(figsize=(7, 4))
# plt.plot(time, x)
# plt.show()
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# d2l.plt.show()

tau = 4
# T-tau样本数，tau就是特征数，因为我们任务当前的yt与前面tau个相关
features = torch.zeros((T - tau, tau))  # 行:样本数，列：特征数
for i in range(tau):
    features[:, i] = x[i:T - tau + i]  # 截取的是x中的从i行开始，总共T-tau行
labels = x[tau:].reshape((-1, 1))  # 样本就是从xtau一直到最后，因为每一个x都是从它之前的tau个数据中得到的
batch_size, n_train = 16, 600


# from d2l.load_array(),将我们的一个数组转变为可以迭代的
def load_array(data_array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


train_iter = load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss:{d2l.evaluate_loss(net, train_iter, loss)}')


net = get_net()
loss = nn.MSELoss()
train(net, train_iter, loss, epochs=30, lr=0.01)

# 预测下一个时间步
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()]
         , 'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# 上面的例子我们使用的都是feature直接做的预测，但是这里我们是根据预测的结果来做的预测
# 所以会导致的结果就是误差在逐渐增大，
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()],
         'time', 'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# 然后通过序列预测k=1，4，16，64
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau):
    features[:, i] = x[i:i + T - tau - max_steps + 1]

for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.show()
