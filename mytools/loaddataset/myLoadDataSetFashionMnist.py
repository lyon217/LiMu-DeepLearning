import torchvision
from torch.utils import data
from torchvision import transforms
import os


def get_dataloader_workers():
    return 0


def my_load_data_fashion_mnist(batch_size, resize=False):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="F:\py\LiMU_DL\dataset", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="F:\py\LiMU_DL\dataset", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
