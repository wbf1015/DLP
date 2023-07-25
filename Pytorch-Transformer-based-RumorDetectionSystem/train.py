import pickle
import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch import optim
from torchnet import meter
from tqdm import tqdm
from model import *
from makedata import *

# 模型输入参数，需要自己根据需要调整
epochs = 100 # 迭代次数
batch_size = 32 # 每个批次样本大小
embedding_dim = 20 # 每个字形成的嵌入向量大小
output_dim = 2 # 输出维度，因为是二分类
lr = 0.003 # 学习率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# 1.获取训练数据
x_train, y_train, x_test, y_test, vocab_size, label_size, src_max_len, index_word_dic, output_dic = data_loading('train.tsv', 'test.tsv')


# 3.将numpy转成tensor
x_train = torch.from_numpy(x_train).to(torch.int32)
y_train = torch.from_numpy(y_train).to(torch.float32)
x_test = torch.from_numpy(x_test).to(torch.int32)
y_test = torch.from_numpy(y_test).to(torch.float32)

# 4.形成训练数据集
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

# 5.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size,
                                           True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size,
                                          False)

# 6.模型训练
model = Transformer(vocab_size, embedding_dim, output_dim, max_len=src_max_len)

Configimizer = optim.Adam(model.parameters(), lr=lr) # 优化器
# Configimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss() # 多分类损失函数

model.to(device)
loss_meter = meter.AverageValueMeter()
test_loss_meter = meter.AverageValueMeter()

best_acc = 0 # 保存最好准确率
best_model = None # 保存对应最好准确率的模型参数

for epoch in range(epochs):
    model.train() # 开启训练模式
    train_acc_count = 0 # 每个epoch训练的样本数
    train_count = 0 # 用于计算总的样本数，方便求准确率
    loss_meter.reset()
    test_acc_count = 0 # 用于统计测试集上的正确数
    test_count = 0 # 用于统计测试集上的综述
    test_loss_meter.reset()
    
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y

        x_input = x_train.long().contiguous()
        x_input = x_input.to(device)
        Configimizer.zero_grad()
        
        # 形成预测结果
        output_ = model(x_input)
        # 记得把他转成cpu的结果
        output_ = output_.cpu()
        # 计算损失
        loss = criterion(output_, y_train.long().view(-1))
        loss.backward()
        Configimizer.step()
        
        loss_meter.add(loss.item())
        
        # 计算每个epoch正确的个数
        train_acc_count += (output_.argmax(axis=1) == y_train.view(-1)).sum()
        train_count += len(x_train)
        
    # 每个epoch对应的准确率
    train_acc = train_acc_count / train_count
    
    for data in tqdm(test_loader):
        model.eval()
        with torch.no_grad():
            x_test, y_test = data  # 解包迭代器中的X和Y

            x_input = x_test.long().contiguous()
            x_input = x_input.to(device)

            # 形成预测结果
            output_ = model(x_input)
            # 记得把结果转成cpu
            output_ = output_.cpu()

            # 计算损失
            loss = criterion(output_, y_test.long().view(-1))
            test_loss_meter.add(loss.item())

            # 统计正确预测的个数
            test_acc_count += (output_.argmax(axis=1) == y_test.view(-1)).sum()
            test_count += len(x_test)
        model.train()
    test_acc = test_acc_count / test_count
    
    # 打印信息
    print("【EPOCH: 】%s" % str(epoch + 1))
    print("训练损失为%s" % (str(loss_meter.mean)), end='  ')
    print("训练精度为%s" % (str(train_acc.item() * 100)[:5]) + '%')
    print('测试损失为%s' % (str(test_loss_meter.mean)), end=' ')
    print('测试精度为%s' % (str(test_acc.item() * 100)[:5]) + '%')

    # 保存模型及相关信息
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = model.state_dict()
    
    # 在训练结束保存最优的模型参数
    if epoch == epochs - 1:
        # 保存模型
        torch.save(best_model, './best_model.pkl')

