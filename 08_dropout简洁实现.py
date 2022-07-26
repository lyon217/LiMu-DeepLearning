from torch import nn
import torch
from d2l import torch as d2l
import torchvision
from torchvision import transforms
from torch.utils import data

dropout1, dropout2 = 0.2, 0.5
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./dataset", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./dataset", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0))


trainer = torch.optim.SGD(net.parameters(), lr=0.3)
num_epochs, batch_size = 10, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()