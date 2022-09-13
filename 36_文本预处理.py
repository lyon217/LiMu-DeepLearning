# --coding:utf-8--
import collections
import operator
import re
from d2l import torch as d2l


# d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
#                                '090b5e7e70c295757f55df93cb0a180b9691891a')
# d2l.download('time_machine')

def read_time_machine():
    with open('../data/timemachine.txt', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
print(f'# 文本总行数： {len(lines)}')
print(lines[0])
print(lines[10])


# 词元化
def tokenize(lines, token='word'):
    """
    将文本行拆分为单词或字符词元
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误，未知次元类型：' + token)


tokens = tokenize(lines, 'char')
for i in range(11):
    print(tokens[i])


# 词表
# 词元的类型是字符串，而模型需要输入的是数字，因此这种类型不方便模型使用，
# 我们构建一个字典叫做词表，用来将字符串类型的词元映射到从0开始的数字索引中，
# 得到的统计结果我们称之为corpus预料，然后根据每个唯一词元的出现频次，为其分配一个数字索引
# 很少出现的词元通常被移除，这可以降低复杂度
# corpus中不存在或者已经删除的任何词元都将映射到一个特殊的未知词元<unk>
# 我们可以选择增加一个列表，用于保存那些被保留的词元，例如填充词元<pad>序列开始词元<bos>序列结束词元<eos>
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        # 通过下标找token就只需要一个list，但是通过token找下标放到list里的复杂度为n，所以放到dict中
        self.idx_to_token = ['<unk>'] + reserved_tokens  # 初始化idx_to_token,将unk放到第一位，对应下标为0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:  # counter.item每次返回的是键值对，第一个是token，二个是freq
            if freq < min_freq:
                # break  # break? 直接结束？
                continue
            if token not in self.token_to_idx:  # 判断token的键是否在token_to_idx中
                self.idx_to_token.append(token)  # 给当前的token分配一个idx
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 将下标映射到对应的token dict中

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freq(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:20])  # 下标前20的词元及其下标
print(vocab.to_tokens([i for i in range(20)]))  # 下标前20的词元
print({token: time for token, time in vocab.token_freq[:10]})  # 前10个词频最高

# 现在可以将一个文本转换为一个数字索引列表
for i in [0, 10]:
    print('文本：', tokens[i])
    print('索引：', vocab[tokens[i]])


# 整合所有功能
def load_corpus_time_machine(max_token=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_token > 0:
        corpus = corpus[: max_token]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine()
