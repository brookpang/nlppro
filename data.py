# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
import csv
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import pickle
import jieba


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



def get_batch(batch, emb_dim=300,maxlen=500):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    # print batch
    lengths = np.array([np.min([maxlen,len(x)]) for x in batch])
    max_len = np.min([np.max(lengths),maxlen])
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(np.min([len(batch[i]),max_len])):
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


def trans_answer(alternatives):
    # return np.array([['<s>'] +[word for word in sent.split(' ') if word in train_w2v] +['</s>'] for sent in X_text_arr])
    # print alternatives
    # for answers in  alternatives:
    #     for answer in answers:
    #         if answer not in train_w2v:
    #             print('sssss',answer)
    #             print answers

    return np.array([[train_w2v[answer] for answer in answers if answer in train_w2v] for answers in  alternatives ])

def load_classifytraindata(trainf, valf):
    query_id_train=[]
    query_id_val=[]
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
            query_id_train.append(query_id)

        for line in rd2:
            query_id, label, seg_passage, seg_query, dict_alternatives = tuple(
                line)
            X_val_passage.append(seg_passage)
            X_val_query.append(seg_query)
            y_val.append(int(label))
            query_id_val.append(query_id)

    return query_id_train,query_id_val,trans_textlist(X_train_passage),trans_textlist( X_val_passage),trans_textlist( X_train_query), trans_textlist(X_val_query),np.array(y_train), np.array(y_val)


def load_classifytraindata2(trainf, valf):
    query_id_train=[]
    query_id_val=[]
    X_train_passage = []
    X_val_passage = []
    X_train_query = []
    X_val_query = []
    alternatives_train = []
    alternatives_val = []

    with open(trainf, 'rb') as rf1, open(valf, 'rb') as rf2:
        rd1 = csv.reader(rf1, delimiter='\t')
        rd2 = csv.reader(rf2, delimiter='\t')
        for line in rd1:
            query_id, answer, seg_passage, seg_query, alternatives = tuple(
                line)
            X_train_passage.append(seg_passage)
            X_train_query.append(seg_query)
            # y_train.append(int(label))
            if query_id=='24124':
                alternatives_train.append([u'好',u'不是',u'无法确定'])
            else:
                alternatives_train.append(alternatives.replace('\xe3\x80\x80','').decode('utf-8').split('|'))

            query_id_train.append(query_id)

        for line in rd2:

            query_id, answer, seg_passage, seg_query, alternatives = tuple(
                line)
            X_val_passage.append(seg_passage)
            X_val_query.append(seg_query)
            # y_val.append(int(label))
            alternatives_val.append(alternatives.replace('\xe3\x80\x80','').decode('utf-8').split('|'))
            query_id_val.append(query_id)


    # print trans_answer(alternatives_train).shape
    return query_id_train,query_id_val,trans_textlist(X_train_passage),trans_textlist( X_val_passage),trans_textlist( X_train_query), trans_textlist(X_val_query),trans_answer(alternatives_train),trans_answer(alternatives_val)


def encode_answer(answer):
    # 备选答案转码
    no = [u'不', u'没',u'无',u'假',u'否',u'非',u'错','未']
    notsure = [u'无法',u'不清楚',u'不确定']

    for ns in notsure:
        if ns in answer:
            return u'无法确定'
    for n in no:
        if n in answer:
            return u'不可以'
    return u'可以'

# 删除固定的词
def del_commonword(alternatives):
    ret_alternatives = ['','','']
    two_answers = {}
    for i,a in enumerate(alternatives):
        if u'无法' in a:
            ret_alternatives[i]=a
        else:
            two_answers[i]=a

    if len(two_answers)==2:
        vs = two_answers.values()
        a0=list(jieba.cut(vs[0]))
        a1=list(jieba.cut(vs[1]))
        if a0==a1 or vs[0] in vs[1] or vs[1] in vs[0] :
            return alternatives
        if a0[0]==a1[0]:
            del a0[0],a1[0]
        if a0[-1]==a1[-1]:
            del a0[-1],a1[-1]

        if len(a0)==0 or len(a1)==0:
            return alternatives

        ks = two_answers.keys()
        ret_alternatives[ks[0]]=''.join(a0)
        ret_alternatives[ks[1]]=''.join(a1)
    else:
        return alternatives

    return ret_alternatives



def read_data(filepath):
    with open(filepath, 'r') as input_data:
        answers,passages,querys,alternatives = [], [], [] ,[]
        # Ignore the headers on the first line of the file.
        # next(input_data)

        for line in input_data:
            line = line.strip().split('\t')
            queryid = line[0]
            answer = ' '.join(line[1].split()).decode('utf-8') # 去除连续空格
            passage =' '.join(line[2].split()).decode('utf-8') # 去除连续空格
            query = ' '.join(line[3].split()).decode('utf-8') # 去除连续空格
            alternative = line[4].decode('utf-8').split('|')
            # if queryid=='24124':
            #     alternative=[u'好',u'不是',u'无法确定']

            # Each premise and hypothesis is split into a list of words.
            answers.append(answer.rstrip().split())
            passages.append(passage.rstrip().split())
            querys.append(query.rstrip().split())


            # encode_sent=[encode_answer(w) for w in alternative]
            # if len(set(encode_sent))==3:
            #     alternative=encode_sent
            # else:
            #     alternative=del_commonword(alternative)

            alternatives.append(alternative)
            # print querys
            # break

        return {"answers": answers,
                "passages": passages,
                "querys": querys,
                "alternatives":alternatives}

def add_worddict(worddict,counts,num_words=None):
    for i, word in enumerate(counts.most_common(num_words)):
        len_worddict=len(worddict)
        if worddict.has_key(word[0]):
            continue
        else:
            worddict[word[0]]=len_worddict
    return worddict



def build_worddict(data,data_dev, num_words=None):
    # words = []
    # [words.extend(sentence) for sentence in data['passages']]
    # [words.extend(sentence) for sentence in data['querys']]
    # [words.extend(sentence) for sentence in data['alternatives']]

    # counts = Counter(words)
    # if num_words is None:
    #     num_words = len(counts)

    # worddict = {word[0]: i+4
    #             for i, word in enumerate(counts.most_common(num_words))}



    words=[]
    [words.extend(sentence) for sentence in data['passages']]
    [words.extend(sentence) for sentence in data_dev['passages']]
    counts = Counter(words)
    worddict = {word[0]: i+4 for i, word in enumerate(counts.most_common())}

    worddict["_PAD_"] = 0
    worddict["_OOV_"] = 1
    worddict["_BOS_"] = 2
    worddict["_EOS_"] = 3

    # print('len_worddict',len(worddict))

    words=[]
    for sentence in data['alternatives']+data_dev['alternatives']:
        words.extend(sentence)
    print('cnt alternatives,{}'.format(len(words)))
    with open('corpus/answer_dict.txt','wb') as wf:
        for w in set(words):
            wf.write(w+'\n')
    # [words.extend(sentence) for sentence in data['alternatives']]
    # [words.extend(sentence) for sentence in data_dev['alternatives']]
    # counts = Counter(words)
    # print(len(data['passages']),len(data['alternatives']),len(words),len(set(words)),len(counts))

    [words.extend(sentence) for sentence in data['querys']]
    [words.extend(sentence) for sentence in data_dev['querys']]


    counts = Counter(words)
    worddict = add_worddict(worddict,counts)

    print('len_queryandalternatives',len(counts))
    print('len_worddict',len(worddict))
    # sss
    # words=[]
    # [words.extend(sentence) for sentence in data['alternatives']]
    # counts = Counter(words)
    # worddict = add_worddict(worddict,counts)
    return worddict



def words_to_indices(sentence, worddict):
    indices = [worddict["_BOS_"]]
    for word in sentence:
        if word in worddict:
            index = worddict[word]
        else:
            # Words absent from 'worddict' are treated as a special
            # out-of-vocabulary word (OOV).
            index = worddict['_OOV_']
        indices.append(index)
    # Add the end of sentence token at the end of the sentence.
    indices.append(worddict["_EOS_"])

    return indices


def words_to_embeddings(sentence, w2v):
    embs = []
    for word in sentence:
        if word in w2v:
            emb = w2v[word]
            embs.append(emb)
        else:
            print '|'.join(sentence)
    return embs


def transform_to_indices(data, worddict):
    transformed_data = {"answers": [], "alternatives": [], "querys": [], "passages": []}

    train_w2v=load_word2vec()

    for i, passage in enumerate(data['passages']):
        # Ignore sentences that have a label for which no index was
        # defined in 'labeldict'.

        indices = words_to_indices(passage, worddict)
        transformed_data["passages"].append(indices)

        indices = words_to_indices(data["querys"][i], worddict)
        transformed_data["querys"].append(indices)

        indices = words_to_embeddings(data["alternatives"][i], train_w2v)
        transformed_data["alternatives"].append(indices)

    return transformed_data


def build_embedding_matrix(worddict):
    train_w2v =load_word2vec()
    vacab_len,vec_size=get_vacabandsize()

    embeddings = {}

    num_words = len(worddict)
    embedding_dim = vec_size
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # Actual building of the embedding matrix.
    for word, i in worddict.items():
        if word in embeddings:
            embedding_matrix[i] = np.array(train_w2v[word], dtype=float)
        else:
            if word == "_PAD_":
                continue
            # Out of vocabulary words are initialised with random gaussian
            # samples.
            embedding_matrix[i] = np.random.normal(size=(embedding_dim))

    return embedding_matrix

def preprocess_NLI_data(inputdir,
                        embeddings_file,
                        targetdir,
                        num_words=None):

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = get_data_path()+'oqmrc_trainingset_classify2.csv'
    dev_file = get_data_path()+'oqmrc_validationset_classify2.csv'

    # -------------------- Train data preprocessing -------------------- #
    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    data = read_data(train_file)
    data_dev=read_data(dev_file)

    print("\t* Computing worddict and saving it...")
    worddict = build_worddict(data,data_dev, num_words=num_words)
    # print worddict
    with open(os.path.join(targetdir, "worddict.pkl"), 'wb') as pkl_file:
        pickle.dump(worddict, pkl_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform_to_indices(data, worddict)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    data = read_data(dev_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform_to_indices(data, worddict)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = build_embedding_matrix(worddict)
    with open(os.path.join(targetdir, "embeddings.pkl"), 'wb') as pkl_file:
        pickle.dump(embed_matrix, pkl_file)



if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Preprocess an NLI dataset')
    parser.add_argument('--config',
                        default="config/preprocessing.json",
                        help='Path to a configuration file for preprocessing')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as cfg_file:
        config = json.load(cfg_file)

    preprocess_NLI_data(os.path.normpath(config["data_dir"]),
                        os.path.normpath(config["embeddings_file"]),
                        os.path.normpath(config["target_dir"]),
                        num_words=config["num_words"])
