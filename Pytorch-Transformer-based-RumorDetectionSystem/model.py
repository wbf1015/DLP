import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch,因为输入的时候肯定是给一个三维的，所以干脆现在就扩展到三维
        # 所以现在的形状大小就是[1, max_len, dmodel]
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class, feedforward_dim=128, num_head=2, num_layers=3, dropout=0.1, max_len=128):
        super(Transformer, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置编码层
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)
        # 编码层
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, feedforward_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        # 输出层
        self.fc = nn.Linear(embedding_dim, num_class)
    
    def forward(self, x):
        # 输入的数据维度为【批次，序列长度】，需要交换因为transformer的输入维度为【序列长度，批次，嵌入向量维度】
        x = x.transpose(0, 1) 
        # 将输入的数据进行词嵌入，得到数据的维度为【序列长度，批次，嵌入向量维度】
        # 就是这里把他理解为一个把二维变三维就可以了，这里和那种linear还不太一样
        x = self.embedding(x)
        # 维度为【序列长度，批次，嵌入向量维度】
        x = self.positional_encoding(x)
        # 维度为【序列长度，批次，嵌入向量维度】
        x = self.transformer(x)
        # 将每个词的输出向量取均值，也可以随意取一个标记输出结果，维度为【批次，嵌入向量维度】
        x = x.mean(axis=0)
        # 进行分类，维度为【批次，分类数】
        x = self.fc(x)
        return x



if __name__ == "__main__":
    # 初始化Shape为(max_len, d_model)的PE (positional encoding)
    max_len = 128
    d_model = 512  
    pe = torch.zeros(max_len, d_model)
    # 初始化一个tensor [[0, 1, 2, 3, ...]]
    position = torch.arange(0, max_len).unsqueeze(1)
    # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )
    # 计算PE(pos, 2i)
    pe[:, 0::2] = torch.sin(position * div_term)
    # 计算PE(pos, 2i+1)
    pe[:, 1::2] = torch.cos(position * div_term)
    # 为了方便计算，在最外面在unsqueeze出一个batch
    pe = pe.unsqueeze(0)
    print(pe.shape)