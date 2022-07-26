import torch
import torchvision
from d2l.torch import Accumulator
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l
from torch import nn


def get_dataloader_workers():
    return 0


def load_data_fashion_mnist(batch_size, resize=None):
    # trans = [transforms.ToTensor()]
    # if resize:
    #     trans.insert(0, transforms.Resize(resize))
    # trans = transforms.Compose(trans)

    trans = transforms.Compose([transforms.ToTensor()])

    mnist_train = torchvision.datasets.CIFAR10(root="./datasets/",
                                               train=True,
                                               transform=trans,
                                               download=False)
    mnist_test = torchvision.datasets.CIFAR10(root="./datasets/",
                                              train=False,
                                              transform=trans,
                                              download=False)
    return (data.DataLoader(mnist_train,
                            batch_size,
                            shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,
                            batch_size,
                            shuffle=False,
                            num_workers=get_dataloader_workers()))


# 定义softmax操作
def softmax(X):
    # 对X中的每一个元素做exp
    X_exp = torch.exp(X)
    # 按照每一行进行求和,得到该行的
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


# 这里是一个例子,我们创建了一个数据样本y_hat,其中包含了2个样本在三个类别的预测概率,以及它们对应的标签y,有了y
# 我们知道在第一个样本中,第一类是正确的预测,而在第二个样本中,第三类是正确的预测,
# 然后我们使用y作为y_hat中概率的索引,我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# cross_entropy(y_hat, y)


# 首选,如果y_hat是矩阵,这里我们假设第二个维度存储每个类的预测分数,我们使用argmax获得每行中最大元素的索引来获得
# 预测类别,然后哦我们将预测类别与真是y元素进行比较,
# 由于等式运算符"=="对数据类型很敏感,因此我们将y_hat与y的类型转化为与y的数据类型一致,
# 结果是一个包含0和1的张量,然后我们求和得到正确预测的数量.
#  序号 类1概率 类2概率 类3概率
# [[1] [0.1    0.2    0.7],
#  [2] [0.2    0.2    0.6]]
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 取每一行的最大值的下标,([序号]可以忽略)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 得到最终预测争取的准确率
# print(accuracy(y_hat, y) / len(y))


# 0.5


# 同样,对于任意数据迭代器data_iter可访问的数据集,我们可以评估在任意模型net的精度
def evaluate_accuracy(net, data_iter):
    # 判断我们使用的是不是torch.nn的模型
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模式设置为评估模式
    metric = Accumulator(2)  # 正确预测数,预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 这里定义一个Accumulator,用于对多个变量进行累加,在上面的evaluate_accuracy函数中,我们在 Accumulator实例中
# 创建了2个变量,分别用于存储正确预测的数量和预测的总数量,当我们遍历数据集时,两者都将随着时间的推移而累加.

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 如果我们使用随机的权重初始net模型,因此该模型的精度应接近于随即猜测,例如在10类的情况下,精度为0.1
# evaluate_accuracy(net, test_iter)


# 我们定义一个函数来训练一个迭代周期,请注意,updater是更新模型参数的常用函数,
# 他接受批量大小作为参数,它可以是d2l.sgd函数也可以是框架的内置优化函数
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# 展示训练函数的实现之前,定义一个再动画中绘制数据的实用程序类Animator
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g--', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# 训练函数, 它会在train_iter访问到的训练数据集上训练一个模型net,该函数将会运行多个周期(由num_epochs指定)
# 在每个迭代周期结束时,利用test_iter访问到的测试数据集对模型进行评估,并且利用Animator来进行可视化训练进度
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert 0.7 < train_acc <= 1, train_acc
    # assert 0.7 < test_acc <= 1, test_acc


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def predit_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    print(type(train_iter))

    # 初始化模型参数
    # num_inputs = 784
    num_inputs = 32 * 32 * 3
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predit_ch3(net, test_iter)