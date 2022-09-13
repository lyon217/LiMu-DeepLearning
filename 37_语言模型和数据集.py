# --coding:utf-8--
from mytools.loaddataset import myLoadTimeMachine
import random
import torch
from d2l import torch as d2l

tokens = myLoadTimeMachine.tokenize(myLoadTimeMachine.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = myLoadTimeMachine.Vocab(corpus)
print(vocab.token_freqs[0:10])
# 可以看到最常出现的其实就是一些虚词，这些词也叫做 stop words停用词

freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='linear', yscale='linear')
# d2l.plt.show()
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
# d2l.plt.show()

# 只能说 这个表达式很python，如此就构建了前后两个词的所有组合
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = myLoadTimeMachine.Vocab(bigram_tokens)

# [(('of', 'the'), 309), (('in', 'the'), 169), (('i', 'had'), 130), ...
print(bigram_vocab.token_freqs[:10])
# 这里值得注意的是，在十个最频繁的词对中，有九个是由两个停用词组成的，只有一个与the time相关，
# 所以我们可以在进一步看看三元语法的频率是否表现出相同的行为：

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = myLoadTimeMachine.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])
# [(('the', 'time', 'traveller'), 59), (('the', 'time', 'machine'), 30), (('the', 'medical', 'man'), 24),
# (('it', 'seemed', 'to'), 16), (('it', 'was', 'a'), 15), (('here', 'and', 'there'), 15),

# 最后我们直观的对比三种模型中的词元频率：
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token:x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.show()


# 随机采样
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引列表
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，来自两个相邻的，随机的，小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机其实索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


# 测试：我们生成一个0-34的序列，假设批量大小为2，时间步数为5，这意味着可以生成 bottom((35-1)/5)=6个‘特征-标签’子序列对
# 假设mini批量大小为2，我们只能得到3个小批量
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X:', X, '\nY:', Y)


# 顺序分区
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个mini batch序列"""
    # 从偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


# 将两个采样函数包装到一个类中，以便稍后可以将其作用数据迭代器
class SeqDataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = myLoadTimeMachine.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_iter_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光及其数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab