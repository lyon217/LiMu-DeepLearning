# --coding:utf-8--
import torch
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 精简输出精度
torch.set_printoptions(2)


# 输入图像的高宽为h，w那么按照缩放比为s（0.1],宽高比为r>0的比例来讲，
# 那么锚框的宽度和高度分别是 ws*sqrt(r),hs/sqrt(r)，
# 带根号r的原因是因为 这样让锚框的宽高比是wsr 与原始输入图像的比例正好是r

# 要生成不同形状的锚框，让我们设置许多缩放比scale取值s1，s2，。。。。。sn，
# 和许多宽高比，aspect ratio，r1，r2,r3....rm
# 当使用这些比例的所有组合来生成锚框时，输入图像总共有whnm个锚框，太多了
# 所以我们只考虑包含s1或r1的组合
# (s1,r1),(s1,r2),...(s1,rm),(s2,r1),(s3,r1),...(sn,r1)
# 也就是说，以同一像素为中心的锚框的数量是n+m-1,对于整个输入图像，我们将共生成wh(n+m-1)个锚框

# 上述方法将在下面的multibox_prior函数中实现，我们指定输入图像， 尺寸列表和宽高列表，然后返回所有的锚框
def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    # 在x和y轴上缩放步长
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    # reshape(-1)就是将张量展为1维
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    # torch.cat()默认dim=0
    w = torch.cat(
        (
            size_tensor * torch.sqrt(ratio_tensor[0]),
            sizes[0] * torch.sqrt(ratio_tensor[1:])
        )
        , ) * in_height / in_width  # 处理矩形输入
    h = torch.cat(
        (
            size_tensor / torch.sqrt(ratio_tensor[0]),
            sizes[0] / torch.sqrt(ratio_tensor[1:])
        ), )  # * in_width / in_height
    # 这里： d2l-book的源码中写的只有w处理了矩形输入，只处理一边也是合理的
    # 如果两边同时进行拉伸，可能会将原本比较宽的图的锚框拉的比较高

    # 除以2来获得半高和半宽, torch.stack(),默认dim=0
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    # 第一次stack的dim=0，得到的是4行的张量，然后.T得到4列的张量，然后将其按照行复制h*w次，列不动
    # 最终得到的行数应该是size(in_width)或size(in_height)*boxes_per_pixel

    # 每个中心点都将有boxes_per_pixel个锚框
    # 所有生成含所有锚框中心的网络，重复了boxes_per_pixel次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1) \
        .repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    # 在第一个维度上扩展一维，应该是batch_size的维度
    return output.unsqueeze(0)


img = d2l.plt.imread('./dataset/catdog.jpg')
h, w = img.shape[:2]
print(h, w)
# 558 729
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)
# torch.Size([1, 2033910, 4])
# 558*729*(3+3-1)=2033910

# 将Y的shape从([1, 2033910, 4])展开为(h,w,5,4),其实也就是将2033910展开为h * w * 5
boxes = Y.reshape(h, w, 5, 4)
print(boxes.shape)  # torch.Size([558, 729, 5, 4])
print(boxes[250, 250, 0, :])  # tensor([-0.15,  0.16,  0.83,  0.74])


# 附加函数------------------------------------------------------------------------------------
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.

    Defined in :numref:`sec_bbox`"""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


# ------------------------------------------------------------------------------------

# 为了显示以图像中某个像素为中心的所有锚框，我们定义了下面的show_bboxes函数来在图像上绘制多个边界框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示bboxes中的所有锚框

    Args:
        bboxes:所有的坐标值，因为要使用的plt.Rectangle()不是一个缩放比例，而是坐标值
            ，所以传入前应该先展开，
    """

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)

        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


d2l.set_figsize(figsize=(10, 10))
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)


# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
#             ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
#              's=0.75, r=0.5'])
# plt.show()


# IOU - intersection over union - 交并比-----------------------------------------------


def box_iou(boxes1, boxes2):
    """
    计算两个锚框或边界框列表中成对的交并比
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              boxes[:, 3] - boxes[:, 1])

    # boxes1,boxes2,areas1,areas2的形状：
    # boxes1：(boxes1的数量，4)，
    # boxes2：(boxes1的数量，4)，
    # areas1：(areas1的数量，)，
    # areas2：(areas2的数量，)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts, inter_lowerrights, inters的形状：
    # (boxes1的数量，boxes2的数量2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areas and union_areas的形状：(boxes1的数量，boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


# 将真实边界框分配给锚框
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框

    Args:
        ground_truth:,
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框的IOU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量，
    # 注意这里full的是anchors的大小，所以再进行赋值的时候，就是要将gt的下标往anchors_bbox_map里填，如下221行
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # 根据阈值，决定是否分配真实边界框

    # dim=1就是在列的维度上进行比较，得到的就是每一行的最大值
    max_ious, indices = torch.max(jaccard, dim=1)

    # jaccard中每一行中的最大值如果大于iou的threshold，那么就拿到他们行数的indexs，nonzero()得到的事下标
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)

    # jaccard中每一行中的最大值如果大于iou的threshold，那么就拿到他们该行中最大值的下标
    box_j = indices[max_ious >= iou_threshold]

    # 将jaccard中的行最大值的下标与行下标对应起来，此时所有下标在jaccard中对应的值都是>=threshold的
    anchors_bbox_map[anc_i] = box_j

    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        # argmax无dim时选择的就是整个张量中的最大值的下标，而且是原始张量.reshape(-1)之后的下标
        max_idx = torch.argmax(jaccard)

        # 原始的jaccard是二维，所以可以直接取余和除得到在jaccard矩阵中对应位置的行和列
        # 注意box_idx对应的是列下标,anc_idx对应的是行下标
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()

        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

        # 通过代码会发现：anchors_bbox_map被赋值了两次，
        # (注：jaccard中存储的每一行代表的是一个锚框和所有的gt的IOU，)
        # 第一次的赋值就是选择了每一行的一个最大值，也就是找到每一个锚框最对应的gt，然后将二者的下标关联起来，存储到abm里
        # 第二次的赋值就是通过一个for循环来循环gt_num次，每次找一个jaccard的矩阵中的最大值然后将该值的行列下标进行关联
        # 第二次的赋值看似和第一次没有任何区别，因为全局的最大值一定是该最大值所在行的最大值，
        # 但是这一步其实是将max_ious >= iou_threshold这个条件给覆盖掉了，
        # 因为有些物体可能特别大，那么很有可能他的IOU就会小于0.5所以此时就通过第二次的赋值来将该gt关联到一个anchor上
        #
        # 那还要第一次赋值有什么用呢？
        # 第二次赋值时的策略和第一次的不同，因为第二次再找到一个最大值赋值了之后，会将该行和该列进行消除，也就是col_discard和row_discard
        # 这就会造成可能另外的行(假设为第L行)的最大值正好在该列上，虽然可能没有当前的最大值大，
        # 所以如果只进行第二次赋值就可能导致:
        # 情况1:要么最终锚框L就没能关联到任何一个gt，(就是被误认为是背景)
        # 情况2：锚框L即使有值，也不是当前行最大的值，也就是没能和最匹配的gt进行关联
    return anchors_bbox_map


# 标记类别和偏移量--------------------------------------------------------------------------------------
# 有了锚框的中心点和wh还有gt的中心点和wh，就可以用公式来衡量一个偏移量
# 一个常见的变换：

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
        对锚框偏移量的转换
    """

    # 两角表示法转换为中心宽度表示法
    center_anc = d2l.box_corner_to_center(anchors)
    center_assigned_bb = d2l.box_corner_to_center(assigned_bb)

    # 两角表示法通常的表示方法是两个坐标值，左下到右上
    offset_xy = 10 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
    offset_wh = 5 * np.log(eps + center_assigned_bb[:, 2:] / anchors[:, 2:])
    # 沿着列的方向进行堆叠，
    offset = np.concatenate([offset_xy, offset_wh], axis=1)
    return offset


# 如果一个锚框没有被分配真是边界框，那么就可以认为是背景，也就是我们说的负类锚框，背景类别的索引设置为0
def multibox_target(anchors, gt_labels):
    """
    使用真实边界框标记锚框

    Args:
        anchors:锚框  shape: batch*num*坐标
        gt_labels:真实边界框 shape:batch*num*(类别+坐标)
    """
    # squeeze的dim属性如果不为1，则张量不会变化，如果anchors的dim=0处为1，则可以squeeze，否则不变
    # 但是下面的num_anchors = anchors.shape[0]直接取了第一个位，那么可能传入参数的时候anchors第一位就是1
    batch_size, anchors = gt_labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]

    for i in range(batch_size):
        gt_label = gt_labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(gt_label[:, 1:], anchors, device)
        # unsqueeze(-1)就是在shape的最后一个位置添加一个1，
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        # 将类标签和分配的边界框坐标初始化为0
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        # 使用真实边界框来标记锚框的类别
        # 如果一个锚框没有被分配，那我们标记为背景，值为0
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = gt_label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = gt_label[bb_idx, 1:]

        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return bbox_offset, bbox_mask, class_labels


# 一个具体例子：
# 图像中定义了狗和猫的gt，第一个元素是类别（0代表狗，1代表猫），其余四个元素是左上角和右下角的（x，y）轴坐标（0，1）之间
# 我们还构建了5个anchor，用左上角和右下角的坐标进行标记A0...A4(索引从0开始)然后我们在图像中回执这些gt和anchors

ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3],
                        [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98],
                        [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
plt.show()

# 使用上面定义的multibox_target函数，我们可以根据猫和狗的真是边界框来标注这些锚框的分类和偏移量
# 背景0，狗1，猫2，我们先为锚框和真是边界框样本添加一个维度
# anchors.shape->[5,4] ground_truth.shape->[2,5]
labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))

print(labels[2])  # tensor([[0, 1, 2, 0, 2]])

# tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.]])
print(labels[1])

# tensor([[-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00,  1.40e+00,  1.00e+01,
#           2.59e+00,  7.18e+00, -1.20e+00,  2.69e-01,  1.68e+00, -1.57e+00,
#          -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -5.71e-01, -1.00e+00,
#           4.17e-06,  6.26e-01]])
print(labels[0])

# 使用NMS（非极大值抑制）预测边界框
