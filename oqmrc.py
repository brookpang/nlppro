# -*- coding:utf-8 -*-

import sys
import json
import csv
import jieba
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import codecs
from sklearn.externals import joblib
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from data import encode_answer,del_commonword
"""
这个算是结构预测
解题思路1：分解大法
先把能用模型处理的进行模型处理
将分错的试图继续分解，用规则或者用其他模型
如：此问题有三个答案，
先分析答案的类型，将答案的类型分成几类解决
再分析其内容，将内容的类型再分解,有语气词的（肯定，否定，怀疑）
确定的肯定和否地的应该算成一类问题，剩下无法确定的在进行细分
问题的特征和回答的特征怎么联系起来还有可选答案特征

解题思路2：深度学习

1自动寻找结构
2自动寻找参数


"""

reload(sys)
sys.setdefaultencoding('utf-8')
jieba.load_userdict('corpus/answer_dict.txt')


def get_data_path():
    return "/home/brook/work/nlppro/data/"


def get_feature_path():
    return "/home/brook/work/nlppro/features/"


def get_databasis_sta():
    # 获取基本的数据统计
    f_train = get_data_path() + "ai_challenger_oqmrc_trainingset.json"
    # f_train = get_data_path() + "test.txt"

    dict_alternatives = {}
    dict_answers = {}
    passage_lens=[]
    with open(f_train, 'rb') as rf:
        line = rf.readline()
        while line:
            json_line = json.loads(line)
            passage = json_line['passage']
            query = json_line['query']
            answer = json_line['answer'].strip()
            query_id = json_line['query_id']
            alternatives = frozenset(
                [s.strip() for s in json_line['alternatives'].split('|')])
            # if len(alternatives)==1:
            #     print query_id,json_line['alternatives']
            if dict_alternatives.has_key(alternatives):
                dict_alternatives[alternatives] += 1
            else:
                dict_alternatives[alternatives] = 1

            if dict_answers.has_key(answer):
                dict_answers[answer] += 1
            else:
                dict_answers[answer] = 1
            passage_lens.append(len(passage))
            # if len(passage)>500 :
            #     print passage,answer
            line = rf.readline()

    # 统计长度
    # passage_lens=np.sort(np.array(passage_lens))
    # print passage_lens
    # print(np.max(passage_lens),np.min(passage_lens),np.median(passage_lens),np.percentile(passage_lens,98),np.percentile(passage_lens,5))
    # print(np.sum(passage_lens>350))
    # print(np.sum(passage_lens==19))


    # f_alternatives_sta = get_data_path() + 'alternatives_sta.txt'
    # with open(f_alternatives_sta, 'wb') as wf:
    #     for k, v in dict_alternatives.iteritems():
    #         wf.write("{}\t{}\t{}\n".format('|'.join(k), v, len(k)))

    # f_answers_sta = get_data_path() + 'answers_sta.txt'
    # with open(f_answers_sta, 'wb') as wf:
    #     for k, v in dict_answers.iteritems():
    #         wf.write("{}\t{}\n".format(k, v))


# def encode_answer(answer):
#     # 备选答案转码
#     no = ['不', '没']
#     notsure = ['无法']

#     for ns in notsure:
#         if ns in answer:
#             return 2
#     for n in no:
#         if n in answer:
#             return 0
#     return 1


def del_stopwordslist(segs):
    stopwords = [
        line.strip()
        for line in codecs.open('./corpus/stopwords.txt', 'r', 'utf-8')
        .readlines()
    ]
    segs = [word for word in list(segs) if word not in stopwords]
    return segs


# 将数据转换成可训练的格式
def trans_data(origin_json, classifyf, laterprocessf, corpusf, is_test=False):
    f_train = get_data_path() + origin_json
    f_train_classify = get_data_path() + classifyf
    f_train_laterprocess = get_data_path() + laterprocessf
    f_train_corpus = get_data_path() + corpusf
    with open(f_train, 'rb') as rf, open(f_train_classify, 'wb') as wf1, open(
            f_train_laterprocess, 'wb') as wf2, open(f_train_corpus,
                                                     'wb') as wf3:
        line = rf.readline()
        while line:
            json_line = json.loads(line)
            passage = json_line['passage'].replace('\t', '')
            query = json_line['query'].replace('\t', '')
            answer = json_line['answer'].strip().replace('\t', '')
            query_id = json_line['query_id']
            alternatives = [
                s.strip() for s in json_line['alternatives'].split('|')
            ]
            seg_passage = ' '.join(del_stopwordslist(jieba.cut(passage)))
            seg_query = ' '.join(del_stopwordslist(jieba.cut(query)))

            dict_alternatives = {}
            for a in alternatives:
                answer_code = encode_answer(a)
                dict_alternatives[a] = answer_code

            if len(set(dict_alternatives.values())) == 3:
                label = dict_alternatives[answer]
                wf1.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    query_id, label, seg_passage, seg_query,
                    dict_alternatives))
            else:
                wf2.write(line)

            wf3.write('{}\n'.format(seg_passage))
            wf3.write('{}\n'.format(seg_query))

            line = rf.readline()



# 将数据转换成可训练的格式,加入可选答案
def trans_data2(origin_json, classifyf, laterprocessf, corpusf, is_test=False):
    f_train = get_data_path() + origin_json
    f_train_classify = get_data_path() + classifyf
    f_train_laterprocess = get_data_path() + laterprocessf
    f_train_corpus = get_data_path() + corpusf
    with open(f_train, 'rb') as rf, open(f_train_classify, 'wb') as wf1, open(
            f_train_laterprocess, 'wb') as wf2, open(f_train_corpus,
                                                     'wb') as wf3:
        line = rf.readline()
        while line:
            json_line = json.loads(line)
            passage = json_line['passage'].replace('\t', '')
            query = json_line['query'].replace('\t', '')
            answer = json_line['answer'].replace(' ','').replace('\t', '')
            query_id = json_line['query_id']
            alternatives =json_line['alternatives'].replace(' ','').replace('\xe3\x80\x80','').decode('utf-8').split('|')
            # if query_id==24124:
            #     alternatives=[u'好',u'不是',u'无法确定']
            #     print alternatives
            # 如果可选答案为空填充默认值
            alternatives = ['__NULL__' if len(a)==0 else a for a in alternatives]

            if query_id==24124:
                print '|'.join(alternatives)

            encode_sent=[encode_answer(w) for w in alternatives]
            if len(set(encode_sent))==3:
                alternatives=encode_sent
            else:
                alternatives=del_commonword(alternatives)

            seg_passage = ' '.join(del_stopwordslist(jieba.cut(passage)))
            seg_query = ' '.join(del_stopwordslist(jieba.cut(query)))

            wf1.write('{}\t{}\t{}\t{}\t{}\n'.format(
                query_id,answer, seg_passage, seg_query, '|'.join(alternatives)))

            wf3.write('{}\n'.format(seg_passage))
            wf3.write('{}\n'.format(seg_query))
            wf3.write('{}\n'.format(' '.join(alternatives)))

            line = rf.readline()

# 加载分类训练数据
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
            y_train.append(float(label))

        for line in rd2:
            query_id, label, seg_passage, seg_query, dict_alternatives = tuple(
                line)
            X_val_passage.append(seg_passage)
            X_val_query.append(seg_query)
            y_val.append(float(label))

    return X_train_passage, X_val_passage, X_train_query, X_val_query, y_train, y_val


# 获取文章和问题的相似度
def get_sim_qa(passasge_vec, query_vec):
    # tf_passage=TfidfVectorizer(max_features=500)
    tf_passage = CountVectorizer()
    simqa = []
    print(passasge_vec.shape)
    for passage, query in zip(passasge_vec, query_vec):
        try:
            tf_passage.fit([passage])
            sparse_passage = tf_passage.transform([passage])
            sparse_query = tf_passage.transform([query])
            simqa.append(cosine_similarity(sparse_passage, sparse_query)[0, 0])
        except Exception, e:
            simqa.append(0.0)
    print(len(simqa))
    return np.array(simqa).reshape(-1, 1)


def model_tfidf():
    """
    模型:
      基于固定语气词的能直接判断出来的模型
      根据模型，选出相应的词tf-idf 提取出相应的词
    """
    trainf = get_data_path() + 'oqmrc_trainingset_classify.csv'
    valf = get_data_path() + 'oqmrc_validationset_classify.csv'
    X_train_passage, X_val_passage, X_train_query, X_val_query, y_train, y_val = load_classifytraindata(
        trainf, valf)

    print('tfidf begin')
    tv_passage=TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9, use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=70000)
    tv_query=TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9, use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=30000)

    X_train_passage = tv_passage.fit_transform(X_train_passage)
    X_val_passage = tv_passage.transform(X_val_passage)

    X_train_query = tv_query.fit_transform(X_train_query)
    X_val_query = tv_query.transform(X_val_query)

    joblib.dump(tv_passage, './models/tv_passage.pkl')
    joblib.dump(tv_query, './models/tv_query.pkl')
    print('tfidf end')

    print('word2vec begin')

    print('word2vec end')

    # print('word2vec begin')
    # w2v = Word2Vec(X_train_passage, size=100, window=5, min_count=3, workers=4)
    # X_train_passage=w2v.wv
    # print(X_train_passage.shape)

    # print('word2vec end')

    print('cosine begin')
    sim_train = get_sim_qa(X_train_passage, X_train_query)
    sim_val = get_sim_qa(X_val_passage, X_val_query)
    print('cosine end')

    ## 0.380557
    print('combine feature begin')
    print(X_train_passage.shape)
    print(sim_train.shape)
    # X_train = hstack((X_train_passage,sim_train))
    # X_val = hstack((X_val_passage,sim_val))

    ## 0.418
    # X_train=sim_train
    # X_val=sim_val

    ## 0.380557
    # X_train = X_train_passage
    # X_val = X_val_passage

    # train-merror:0.273466	valid-merror:0.356084
    X_train = hstack((X_train_passage, X_train_query))
    X_val = hstack((X_val_passage, X_val_query))

    print('combine feature end')

    print('mnb begin')
    model_NB = MultinomialNB()
    model_NB.fit(X_train, y_train)
    pred_train = model_NB.predict(X_train)
    pred_val = model_NB.predict(X_val)
    print('the acc is {}'.format(accuracy_score(y_train, pred_train)))
    print('the acc is {}'.format(accuracy_score(y_val, pred_val)))

    print('mnb end')

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_val, label=y_val)

    params = {}
    params["objective"] = "multi:softmax"
    # params["objective"] = "multi:softprob"
    params['num_class'] = 3
    params['booster'] = 'gbtree'
    params["eta"] = 0.1
    params["max_depth"] = 10
    params["silent"] = 1
    params["seed"] = 1024

    watchlist = [(d_train, 'train'), (d_val, 'valid')]
    bst = xgb.train(
        params,
        d_train,
        500,
        watchlist,
        early_stopping_rounds=50,
        verbose_eval=2)
    bst.save_model('./models/xgb_classify.model')

    xgb_pred_train=bst.predict(d_train)
    xgb_pred_val=bst.predict(d_val)

    print('the acc is {}'.format(accuracy_score(y_train, xgb_pred_train)))
    print('the acc is {}'.format(accuracy_score(y_val, xgb_pred_val)))

    # 将分错的数据单独写入文件

    # 统计各个分类比例，包括原始部分比例，分错部分比例

    # feature importance
    feats = tv.get_feature_names()
    indices = range(50000)
    with open(get_feature_path() + 'xgb_classify.fmap', 'wb') as wf:
        for i, f in zip(indices, feats):
            # use i for indicator and q for quantity
            wf.write('{}\t{}\tq\n'.format(i, f))

    importance = bst.get_score(
        fmap=get_feature_path() + 'xgb_classify.fmap', importance_type='gain')
    with open(get_feature_path() + 'xgb_classify_importance.csv', 'wb') as wf:
        for k, v in importance.iteritems():
            wf.write('{}\t{}\n'.format(k, v))


def rule_tail():
    """
    规则:长尾词处理规则
    :对只有一个备选答案的处理规则
    :对只有2个备选答案的处理规则
    """
    pass


def train_combine():
    """
    模型及规则整合
    """
    rule_tail()
    model_tfidf()


def val():
    """
     评估每一次模型总体变化
     评估每条数据的评分变化
    """
    pass


def score():
    """
     评估每一次得分，并记录相应的变化
    """
    pass


def submit():
    """
     提交结果
    """
    pass


if __name__ == '__main__':
    # get_databasis_sta()
    # trans_data('ai_challenger_oqmrc_trainingset.json',
    #            'oqmrc_trainingset_classify.csv',
    #            'oqmrc_trainingset_laterprocess.csv',
    #            'oqmrc_trainingset_corpus.csv')
    # trans_data('ai_challenger_oqmrc_validationset.json',
    #            'oqmrc_validationset_classify.csv',
    #            'oqmrc_validationset_laterprocess.csv',
    #            'oqmrc_validationset_corpus.csv')

    trans_data2('ai_challenger_oqmrc_trainingset.json',
               'oqmrc_trainingset_classify2.csv',
               'oqmrc_trainingset_laterprocess2.csv',
               'oqmrc_trainingset_corpus2.csv')

    trans_data2('ai_challenger_oqmrc_validationset.json',
               'oqmrc_validationset_classify2.csv',
               'oqmrc_validationset_laterprocess2.csv',
               'oqmrc_validationset_corpus2.csv')
    # train_combine()
