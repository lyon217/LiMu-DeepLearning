# --coding:utf-8--
import os
import torch
from d2l import torch as d2l

# d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
#                            '94646ad1522d915e7b0f9296181140edcf86a4f5')
#
#
# def read_data_nmt():
#     """载入'英语-法语'数据集"""
#     data_dir = d2l.download_extract('fra-eng')
#     with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
#         return f.read()


data_dir = r'../data/fra-eng/fra.txt'
with open(data_dir, 'r', encoding='utf-8') as f:
    raw_text = f.read()
print(raw_text[:75])


# 用空格替换不间断空格(non-breaking space)，使用小写字母替换大写字母，并在单词和标点之间插入空格
def preprocess_nmt(text):
    # 判断,.!?符号前是否有空格，没有空格返回True，有返回False
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # \u202f不换行窄空格Narrow No-Break Space , \xa0 是不间断空白符&nbsp;
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 从i=1开始如果当前字符是特殊字符并且字符前没有空格则添加一个空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


text = preprocess_nmt(raw_text)
print(text[:80])


# 在机器翻译中，我们更喜欢单词级词元化 （最先进的模型可能使用更高级的词元化技术）。
# 下面的tokenize_nmt函数对前面num_examples个文本序列对进行词元，其中每个词元要么是一个词，要么是一个标点
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


source, target = tokenize_nmt(text)
print(source[:30], '---------------', target[:30])


# 让我们绘制每个文本序列所包含的词元数量的直方图，这个简单的‘英-法’数据集中，大多数文本序列的次元数量少于20个
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)


show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target)
d2l.plt.show()

# 词表
# 由于机器翻译数据集由语言对组成，因此可以分为源语言和目标语言构建两个词表，
# 使用单词级词元化时，此表大小将明显大于使用字符级词元化时的词表大小
# 为了缓解这一问题，我们将出现次数少于2次的狄品慈元视为相同的未知<unk>词元，除此之外，我们还制定了额外的特定词元
# 例如在小批量时用于将序列填充到相同长度的填充词元<pad>以及序列的开始词元<bos>和结束词元<eos>这些特殊词元在自然任务中比较常用
src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(len(src_vocab))


# 加载数据集
# 语言模型中序列样本都有一个固定的长度，无论这个样本是一个句子的一部分还是跨越了多个句子的一个片段，
# 这个固定的长度是由之前的参数num_steps来指定的，
# 但是机器翻译中，每个样本都是由源和目标组成的文本序列对，其中的每个文本序列可能具有不同的长度
# 为了提高计算效率，我们仍然可以通过截断(truncation)和填充(padding)方式实现一次只处理一个小批量的文本序列，
# 假设同一个小批量中的每个序列都应该具有相同的长度num_steps，那么如果文本序列的词元数目少于num_steps，我们在
# 末尾添加特定词元<pad>词元，直到其长度达到num_steps，反之我们将截断其长度至num_steps，并且丢弃剩余的词元
# 这样每个文本序列都具有相同的长度,以便相同形状的小批量进行加载
def truncate_pad(line, num_steps, pading_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [pading_token] * (num_steps - len(line))


truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])


# 现在我们定义一个函数，可以将文本序列 转换成小批量数据集用于训练。
# 我们将特定的“<eos>”词元添加到所有序列的末尾， 用于表示序列的结束。
# 当模型通过一个词元接一个词元地生成序列进行预测时， 生成的“<eos>”词元说明完成了序列输出工作。
# 此外，我们还记录了每个文本序列的长度， 统计长度时排除了填充词元
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    # sum(1)将对每一行进行统计，因为reduce的是列
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# 训练数据
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据及的迭代器和词表"""
    text = preprocess_nmt(raw_text)
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
