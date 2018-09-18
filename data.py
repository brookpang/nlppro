# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
import csv
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

model_file_name = 'corpus/w2v.model.bin'

def load_word2vec():
    train_w2v = KeyedVectors.load_word2vec_format(model_file_name, binary=True)
    return train_w2v

train_w2v=load_word2vec()
# word_vec=load_word2vec()
def get_vacabandsize():
    # model_file_name = './corpus/w2v.model.bin'
    # train_w2v = KeyedVectors.load_word2vec_format(model_file_name, binary=True)
    return len(train_w2v.vocab),train_w2v.vector_size



def get_batch(batch, emb_dim=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = train_w2v[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths

# 将seq中的词转成向量
def word2vec_seq(seq, word2vec):
    transformed_seq = []
    word_embedding_size = word2vec.syn0.shape[1]
    for word in seq:
        try:
            word_arr = word2vec[word]
            new_word_arr = normalize(word_arr.reshape(1, -1), copy=True)
            transformed_seq.append(new_word_arr.reshape(word_embedding_size))
        except KeyError, e:
            print('keyerror')
    return transformed_seq


def get_data_path():
    return "data/"


# train_w2v = load_word2vec()
def trans_textlist(X_text_arr):
    # return np.array([['<s>'] +[word for word in sent.split(' ') if word in train_w2v] +['</s>'] for sent in X_text_arr])
    return np.array([[word for word in sent.decode('utf-8').split(' ') if word in train_w2v] for sent in X_text_arr])



def load_classifytraindata(trainf, valf):
    X_train_passage = []
    X_val_passage = []
    X_train_query = []
    X_val_query = []
    y_train = []
    y_val = []

    with open(trainf, 'rb') as rf1, open(valf, 'rb') as rf2:
        rd1 = csv.reader(rf1, delimiter='\t')
        rd2 = csv.reader(rf2, delimiter='\t')
        for line in rd1:
            query_id, label, seg_passage, seg_query, dict_alternatives = tuple(
                line)
            X_train_passage.append(seg_passage)
            X_train_query.append(seg_query)
            y_train.append(int(label))

        for line in rd2:
            query_id, label, seg_passage, seg_query, dict_alternatives = tuple(
                line)
            X_val_passage.append(seg_passage)
            X_val_query.append(seg_query)
            y_val.append(int(label))

    return trans_textlist(X_train_passage),trans_textlist( X_val_passage),trans_textlist( X_train_query), trans_textlist(X_val_query),np.array(y_train), np.array(y_val)
