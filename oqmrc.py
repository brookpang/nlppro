# -*- coding:utf-8 -*-

import sys
import json
"""
解题思路：分解大法
先把能用模型处理的进行模型处理
将分错的试图继续分解，用规则或者用其他模型
如：此问题有三个答案，
先分析答案的类型，将答案的类型分成几类解决
再分析其内容，将内容的类型再分解,有语气词的（肯定，否定，怀疑）
问题的特征和回答的特征怎么联系起来还有可选答案特征

"""


reload(sys)
sys.setdefaultencoding('utf-8')

def get_data_path():
    return "/home/brook/work/competition/data/"


def get_alternatives_set():
    f_train = get_data_path() + "ai_challenger_oqmrc_trainingset.json"
    # f_train = get_data_path() + "test.txt"

    dict_alternatives = {}
    with open(f_train, 'rb') as rf:
        line = rf.readline()
        while line:
            json_line = json.loads(line)
            passage = json_line['passage']
            query = json_line['query']
            alternatives = frozenset([s.strip() for s in json_line['alternatives'].split('|')])
            if dict_alternatives.has_key(alternatives):
                dict_alternatives[alternatives] += 1
            else:
                dict_alternatives[alternatives] = 1

            line = rf.readline()
    # print(dict_alternatives)
    for k,v in dict_alternatives.iteritems():
        for alter_answer in k:
            if '不' in alter_answer:

        print "{}\t{}".format('|'.join(k),v)

def train():
    pass


def val():
    """
     评估每一次模型变化，带来的每天数据的评分变化
    """
    pass


def score():
    """
     评估每一次得分，并记录相应的变化
    """
    pass


if __name__ == '__main__':
    get_alternatives_set()
    train()
