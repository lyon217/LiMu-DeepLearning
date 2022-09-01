import torch
from torch import nn
import mytools
from mytools.loaddataset import myLoadDataSetFashionMnist
from d2l import torch as d2l

from mytools import Train

net = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)
                    , nn.ReLU()
                    , nn.MaxPool2d(kernel_size=3, stride=2)

                    , nn.Conv2d(96, 256, kernel_size=5, padding=2)
                    , nn.ReLU()
                    , nn.MaxPool2d(kernel_size=3, stride=2)

                    , nn.Conv2d(256, 384, kernel_size=3, padding=1)
                    , nn.ReLU()

                    , nn.Conv2d(384, 384, kernel_size=3, padding=1)
                    , nn.ReLU()

                    , nn.Conv2d(384, 256, kernel_size=3, padding=1)
                    , nn.ReLU()
                    , nn.MaxPool2d(kernel_size=3, stride=2)

                    , nn.Flatten()

                    , nn.Linear(256 * 6 * 6, 4096, bias=True)
                    , nn.ReLU()
                    , nn.Dropout(0.5)

                    , nn.Linear(4096, 4096, bias=True)
                    , nn.ReLU()
                    , nn.Dropout(0.5)

                    , nn.Linear(4096, 10, bias=True))

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t ', X.shape)

batch_size = 128
train_iter, test_iter = myLoadDataSetFashionMnist.my_load_data_fashion_mnist(batch_size=batch_size, resize=224)

lr, num_epochs = 0.9, 3
Train.train_ch6(net, train_iter, test_iter, num_epochs, lr, mytools.try_gpu())
d2l.plt.show()