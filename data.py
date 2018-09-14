
import os
import numpy as np
import torch
import csv

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def get_batch(batch, word_vec, emb_dim=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_data_path():
    return "/home/brook/work/nlppro/data/"


def load_word2vec():
    from gensim.models.keyedvectors import KeyedVectors
    model_file_name = './corpus/w2v.model.bin'
    train_w2v = KeyedVectors.load_word2vec_format(model_file_name, binary=True)
    return train_w2v

def trans_textlist(X_text_arr):
    train_w2v = load_word2vec()
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
