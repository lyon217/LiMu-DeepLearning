import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), \
           torch.LongTensor(output_batch), \
           torch.LongTensor(target_batch)


def get_sinusiod_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    # tensor.byte()等同于self.to(torch.uint8)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


# 4.get_attn_pad_mask
# 首先是为什么要有这个函数，其次是如何使用这个函数中的符号矩阵
# 为什么有------------？这个函数的作用就是为了防止在句子长度不够然后使用pad进行填充时，
# pad也会对前面的真实的词产生影响，因为经过attention之后要做softmax，
# softmax中pad部分的影响必须需要是0，所以要有一个符号矩阵来控制，pad的这部分是否要进入softmax进行计算
# 如何使用------------？就是产生一个符号矩阵0代表不是pad，1代表是pad，
# 然后控制当前token是否进入softmax进行计算

# 比如说现在的句子长度为5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmmax之前，我们得到的形状
# 为len_input * len_input 代表每个单词对其余包含自己的单词的影响力
# 所以这里我们需要有一个同等大小的矩阵，告诉我哪个位置是PAD部分，之后再计算softmax之前会把这里设置为无穷大
# 一定要注意的是这里得到的矩阵形状是batch_size * len_q * len_k, 我们对k中的pad符号进行标识
# 并没有对k做标识，因为没必要
# seq_q和seq_k不一定一致，在交互注意力中，q来自解码端，所以告诉模型编码这边pad符号信息就可以
# 解码端的pad信息在交互注意力层是没有用到的
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size * 1 * len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)


# ## 3. PositionalEncoding 代码实现
# class PositionalEncoding(nn.Module):
#     def __init__(self, dim_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#
#         # 位置编码实现其实很简单直接对着公式去敲就可以，下面这个代码只是其中一种实现方式
#         # 从理解来讲，需要注意的就是偶数和奇数在共识上有一个共同部分，我们需要使用log函数把次方拿下来，方便计算：
#         # 假设dim_model=512,那么公式里的pos代表的从0，1，2。。。511的每个位置，2i那个符号从中i从0取到了255，那么2i对应的就是0，2，4。。510
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, dim_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
#         pe[:, 0::2] = torch.sin(position * div_term)  # 这里需要注意的是pe[:, 0::2]这个用法从0开始到最后面，步长为2，代表偶数位置
#         pe[:, 1::2] = torch.cos(position * div_term)
#         # 上面代码获取之后得到的pe: [max_len * dim_model]
#
#         # 下面这个代码之后，我们得到的pe的形状为：[max_len * dim_model]
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)  # 定一个缓冲区，其实简单理解为这个参数不更新就可以
#
#     def forward(self, x):
#         """
#         Args:x:[seq_len, batch_size, dim_model]
#         """
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射的到参数矩阵Wq,Wk,Wv
        self.W_Q = nn.Linear(dim_model, d_k * n_heads)
        self.W_K = nn.Linear(dim_model, d_k * n_heads)
        self.W_V = nn.Linear(dim_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=dim_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=dim_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, inputs):
        residual = inputs  # inputs:[batch_size, len_q, dim_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


## 5.EncoderLayer： 包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 下面这个就是做自注意力层，输入的是enc_inputs，形状是[batch_size * seq_len_q * dim_model]
        # 需要注意的是最初的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size * len_q * dim_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


# 2.Encoder部分包含三个部分：词向量Embedding，位置编码部分，注意力层以及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 这里其实就是定义生成一个矩阵，大小是src_vocab_size * dim_model
        self.src_emb = nn.Embedding(src_vocab_size, dim_model)

        ## 位置编码情况，这里是固定的正弦余弦函数，也可以使用类似词向量的nn.Emedding获得一个可以更新学习的位置编码
        # self.pos_emb = PositionalEncoding(dim_model)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusiod_encoding_table(src_len + 1, dim_model), freeze=True)

        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # enc_inputs的形状：[batch_size * source_len]
        # 下面通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1, 2, 3, 4, 0]]))

        # 这里是位置编码，然后将前面两者相加放到函数里，这里可以去看位置编码函数实现3.
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响,这里去看函数4.
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []

        for layer in self.layers:
            # 去看EncoderLayer层函数5.
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dim_model)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusiod_encoding_table(tgt_len + 1, dim_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs:[bat_s, target_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  # 编码层
        self.decoder = Decoder()  # 解码层

        # 输出层dim_model是我们解码层每个token输出的维度大小，之后会做一个tgt_vocab_size大小的softmax
        self.projection = nn.Linear(dim_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        Args:
            enc_inputs:形状为[batch_size, src_len]，主要是作为编码端的输入
            dec_inputs,形状为[batch_size, tgt_len],主要是作为解码端的输入
        """

        # enc_outputs就是主要的输出,输出由自己决定，想输出什么输出什么，可以是全部tokens的输出，可以使特定每一层的输出，也可以是中间参数的输出
        # enc_self_attns这里就是QK转置之后softmax之后的矩阵值，代表的是每个单词和其他单词的相关性
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs是decoder主要输出，用于后续的linear映射，
        # dec_self_attens类比于enc_self_attens是每个单词对decoder中输入的其余单词的相关性，
        # dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs做映射到词表大小
        # dec_logits: [batch_size * src_vocab_size * tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


if __name__ == '__main__':
    # 句子的输入部分
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    # 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5
    tgt_len = 5

    # 模型参数
    dim_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # numver of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
