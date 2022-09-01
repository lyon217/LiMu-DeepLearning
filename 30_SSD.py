# --coding:utf-8--
# 单发多框检测SSD
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from mytools import Timer, Animator, Accumulator


# 类别预测层
def cls_predictor(num_inputs, num_anchors, num_classes):
    # 就是预测总像素数*每个像素的anchor数*（每个锚框预测的类别+背景类）
    # 特征图每个像素对应a锚框，每个锚框对应q分类，单个像素就要a*(q+1)个预测信息
    # 这个信息通过卷积核的多个通道来存储，所以这里进行卷积操作
    # 图像分类，只预测分类情况，所以接全连接层，这里单个像素的预测结果太多，就用多个通道来存储
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


# 边界框预测层
def bbox_predictor(num_inputs, num_anchors):
    # 预测锚框的偏移量，通道数变为原来的4倍，分别对应4个值
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


# 两个例子Y1，Y2是feature map的大小
# (2,8,20,20)中，2是batch_size,8是通道数，这个是不变的，8是当前的通道数，20*20是当前的wh，
# (8,5,10)中，8是当前的通道数，需要与上一个8对应，5是每个像素需要生成的锚框数，10是预测的类别数
# 所以类别预测输出中的shape应该[2, 5*(10+1), 20, 20]
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)  # torch.Size([2, 55, 20, 20]) torch.Size([2, 33, 10, 10])


def flatten_pred(pred):
    # permute()返回的是原始tensor的view
    # flatten的start_dim属性是指定shape开始permute的下标
    # flatten函数让通道数放到了最后面，这样的好处是让每一个像素的预测值单拿出来时是一个连续值
    # 这样的话，如果我们取perd[1, 1, 1, :]的话那就是一个像素上的所有的预测值
    # 而且最重要的一点就是，如果不进行permute的话，那么我们进行最终的预测时一旦在fc前进行flatten后
    # 同一像素上的不同通道上的预测值会被拉开，就不是相邻的了，操作起来就非常不方便
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    """
    将两个flatten后的tensor在列上进行合并
    """
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


print(concat_preds([Y1, Y2]).shape)  # torch.Size([2, 25300])  -> 55*20*20 + 33*10*10


# 使得块的channel改变为想要的值，然后大小减半的block
def down_sample_blk(in_channels, out_channels):
    """
    两层卷积，k3p1，使得大小不变，然后channel变为原来的二倍，然后在最后返回的时候，加上一个MaxPooling使大小减半
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# 原始图像batch_size=2,channel=3,大小20*20，然后经过downsample_blk使得channel从3变为10，然后大小减半
print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)  # torch.Size([2, 10, 10, 10])


# 从输入图像一直到我们第一次对feature map进行选择锚框的中间一截
def base_net():
    blk = []
    # 第一个3就是原始图像的channel3，然后后面还有三个数字就是每一次变化的值
    # 3->16，16->32,32->64,通道数最终变为64，然后被down_sample_blk了3次，大小缩小8倍
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)  # torch.Size([2, 64, 32, 32])


# 完整的单发多框检测模型由5个模块组成：
# 其实就是5个stage
# 这里每个stage的blk可以按照实际情况来写，就像这里stage2和3都是128->128，因为我们的数据集比较小了，所以没必要做到512
# 假设原始图像大小为256，
# 经过stage0后，经过了3个down_sample_blk,此时图像大小变为了256/2/2/2=32,channel变为了64，
# 经过stage1后，经过了1个down_sample_blk,此时图像大小变为了16，        ,channel变为了128
# ........2........1..................................8,.............没变，仍为128
# ........3........1..................................4,.............没变，仍为128
# ........4..,经过了一个AdaptiveMaxPool2d，图像大小变为了1，..............没变，仍为128
# 那么生成的锚框数为：(32*32 + 16*16 + 8*8 + 4*4 + 1) * 4=5444个锚框
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


# 为每个块定义前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """
    如果是卷积块的，其实就是forward
    主要是要处理anchors

    Args:
        X: 输入图像的tensor
        blk：网络
        size，ratio： 当前feature map尺度下，我的anchor是什么样子的，然后生成出来
        class_predictor,bbox_predictor: 就是我们构建的当前blk下的预测
    """
    Y = blk(X)
    # 在当前尺度下，
    # 这里anchors的生成可以写到函数外，因为锚框的生成跟Y的值没有关系，只跟Y的尺寸有关系
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


sizes = [[0.20, 0.272],  # 0.2+0.17+0.17+0.17.....
         [0.37, 0.447],  # 0.272=sqrt(0.2*0.37)
         [0.54, 0.619],  # 0.447=sqrt(0.37*0.54)
         [0.71, 0.790],  # ...
         [0.88, 0.961]]  # 最一开始的尺寸比较小，看的是小物体，后面尺寸大，看的是尺寸大的物体
ratios = [[1, 2, 0.5]] * 5  # 1，2，0.5就是常用组合, 得到的是一个二维数组，不是三维的
# 每一个像素我们生成4个锚框
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


# 定义完整模型 TinySSD:
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        """
        Args:
            num_classes: 类别数
        """
        super(TinySSD, self).__init__()
        self.num_classes = num_classes

        # 每个stage对应的channel数
        idx_to_in_challens = [64, 128, 128, 128, 128]
        for i in range(5):  # 5个stage
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_challens[i],
                                                    num_anchors,
                                                    num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_challens[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'bli_%d'%i)即可访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))

        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0],
                                      -1,
                                      self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# 我们创建一个模型实例，然后使用它对一个256*256像素的小批量图像X执行前向传播
net = TinySSD(num_classes=1)
# 批量大小32，通道3，图像大小256*256
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

# 锚框数总共5444个，每个锚框4个坐标，
print('output anchors:', anchors.shape)  # output anchors: torch.Size([1, 5444, 4])
# [32, 5444, 2]中的2是2类，一类是物体类（我们的简单例子中只有香蕉分类），一类是背景类
print('output chass preds:', cls_preds.shape)  # output chass preds: torch.Size([32, 5444, 2])
# [32, 21776]中的21776=5444*4，就是每个锚框要有4个偏移量
print('outout bbox preds:', bbox_preds.shape)  # output bbox preds: torch.Size([32, 21776])

batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size=batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 有些锚框的预测是非常不靠谱的，所以使用L1的好处就是即使预测的非常不靠谱，那么也不会得到一个非常大的Loss
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(
        cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)
    ).reshape(batch_size, -1).mean(dim=1)
    # bbox_masks的作用就是如果不是背景类那就是1，是背景类就是0，因为背景类是没有收敛目标的
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


# 准确率评价
def cls_eval(cls_preds, cls_labels):
    return float(
        (
                cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels
        ).sum()
    )


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float(
        (
            torch.abs((bbox_labels - bbox_preds) * bbox_masks)
        ).sum()
    )


# 训练模型
num_epochs, timer = 20, Timer.Timer()
animator = Animator.Animator(xlabel='epoch', xlim=[1, num_epochs],
                             legend=['class error', 'bbox mae'])

net = net.to(device)
for epoch in range(num_epochs):
    metric = Accumulator.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)

        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # print(anchors.device, cls_preds.device, bbox_preds.device)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        timer.stop()
        metric.add(cls_eval(cls_preds, cls_labels),
                   cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')
print(f'All time: {timer.sum()}')
d2l.plt.show()

# 预测目标
X = torchvision.io.read_file('../').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()


# 使用multibox_detection函数，可以根据锚框及其预测偏移量得到预测边界框，然后通过NMS来移除相似边界框
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != 1]
    return output[0, idx]


output = predict(X)


# 筛选置信度>=0.9的边界框作为最终的输出
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


display(img, output.cpu(), threshold=0.9)