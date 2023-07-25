import pickle
import numpy as np
import pandas as pd
import random
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def get_max_len(sentences):
    max_len = 0
    for sentence in sentences:
        if len(sentence) > max_len:
            max_len = len(sentence)
    
    return max_len

def data_loading(train_path, test_path):
    # 我觉得大家需要共享一个数据字典,所以同时拿到训练接和测试集的所有数据
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    train_label, train_vocabulary = list(train_df['label'].unique()), list(train_df['text'].unique())
    test_label, test_vocabulary = list(test_df['label'].unique()),  list(test_df['text'].unique())
    labels = list(set((train_label + test_label)))
    vocabulary = list(set((train_vocabulary + test_vocabulary)))
    max_len = get_max_len(vocabulary)

    vocab_list = []
    for sentence in vocabulary:
        for word in sentence:
            vocab_list.append(word)
    vocab_list = list(set(vocab_list))

    # 这样的话我可以把读出来的东西都打乱一下
    random.shuffle(vocab_list)

    # 相当于做一个one_hot编码
    word_index_dic = {word: i + 1 for i, word in enumerate(vocab_list)}
    # Python中的 pickle 模块实现了基本的数据序列与反序列化。序列化对象可以在磁盘上保存对象，并在需要的时候读取出来
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_index_dic, f)
    index_word_dic = {i + 1: word for i, word in enumerate(vocab_list)}

    # 感觉这一块有点多此一举，但是当label是一些其他奇奇怪怪的东西的时候这么写也没问题
    label_dic = {label: i for i, label in enumerate(labels)}
    with open('label_dict.pk', 'wb') as f:
        pickle.dump(label_dic, f)
    # idx2label 将0和1映射为正反面
    output_dic = {i: labels for i, labels in enumerate(labels)}

    # 这里加一是因为还需要把零加进去
    vocab_size = len(word_index_dic.keys()) + 1
    label_size = len(label_dic.keys())
    
    train_sentence = [[word_index_dic[word] for word in sentence] for sentence in train_df['text']] 
    train_label = [[label_dic[label]] for label in train_df['label']]
    test_sentence = [[word_index_dic[word] for word in sentence] for sentence in test_df['text']]
    test_label = [[label_dic[label]] for label in test_df['label']]
    
    train_sentence = pad_sequences(maxlen=max_len, sequences=train_sentence, padding='post', value=0)
    test_sentence = pad_sequences(maxlen=max_len, sequences=test_sentence, padding='post', value=0)
    
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    
    # print(train_sentence.shape)
    # print(test_sentence.shape)
    # print(type(train_sentence))
    return train_sentence, train_label, test_sentence, test_label, vocab_size, label_size, max_len, index_word_dic, output_dic

    

if __name__ == "__main__":
    data_loading('train.tsv', 'test.tsv')