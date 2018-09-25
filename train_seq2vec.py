# -*- coding:utf-8 -*-
"""
  transformatted sentence to fixed size vec
"""
import os
# import gensim.models.word2vec as w2v
# from gensim.test.utils import datapath
# from gensim.models.keyedvectors import KeyedVectors
# from sklearn.preprocessing import normalize

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from nnmodel import NLINet
from data import get_vacabandsize
from data import load_classifytraindata, get_data_path, get_batch

# model_file_name = './corpus/w2v.model.bin'

parser = argparse.ArgumentParser(description='NLi training')
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--output", type=str, default='models/', help="Output model directory")
parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')

args=parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print args.cuda
print args.n_epochs

"""
SEED
"""
seed=2048
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

"""
train
"""
vacab_len,vec_size=get_vacabandsize()

params = {
    'n_words': vacab_len,
    'word_emb_dim': vec_size,
    'enc_lstm_dim': 128,
    'n_enc_layers': 1,
    'dpout_model': 0.5,
    'dpout_fc': 0.5,
    'fc_dim': 32,
    'bsize': 64,
    'n_classes': 3,
    'pool_type': 'max',
    'nonlinear_fc': 1,
    'use_cuda': False,
    'optimizer': 'adam'
}

nli_net = NLINet(params)
print(nli_net)
outputdir = args.output
outputmodelname = 'nli.pickle'
outputmodelnamecuda = 'nlicuda.pickle'
lrshrink = 5.
minlr = 1e-5

# loss
weight = torch.FloatTensor(params['n_classes']).fill_(1)
# weight = torch.FloatTensor([0.3,0.16,1.])
loss_fn = nn.CrossEntropyLoss(weight=weight)
# loss_fn = nn.MultiMarginLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optimizer = optim.Adam(nli_net.parameters())

# cuda by default
if args.cuda:
    nli_net.cuda()
    loss_fn.cuda()
"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
max_norm = 5.
lr = None
n_epochs = args.n_epochs
decay = 0.99
n_earlystopping=0

# trainf = get_data_path() + 'oqmrc_trainingset_classify.csv'
trainf = get_data_path() + 'trainingset_classify_filter.csv'
valf = get_data_path() + 'oqmrc_validationset_classify.csv'
query_id_train,query_id_val,X_train_passage, X_val_passage, X_train_query, X_val_query, y_train, y_val = load_classifytraindata(trainf, valf)

def get_permutation(x):
    index_0=np.argwhere(x==0).flatten()
    index_1=np.argwhere(x==1).flatten()
    index_2=np.argwhere(x==2).flatten()

    len2=len(index_2)

    permutation_0 = np.random.permutation(len(index_0))
    permutation_1 = np.random.permutation(len(index_1))
    permutation_2 = np.random.permutation(len(index_2))

    index_0 = index_0[permutation_0]
    index_1 = index_1[permutation_1]
    index_2 = index_2[permutation_2]
    sel_index=np.hstack((index_0[0:len2],index_1[0:len2],index_2))
    permutation_sel = np.random.permutation(len(sel_index))
    return sel_index[permutation_sel]



def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(X_train_passage))
    # permutation = get_permutation(y_train)
    # print(permutation,len(permutation))

    s1 = X_train_passage[permutation]
    s2 = X_train_query[permutation]
    target = y_train[permutation]
    # print target

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decay if epoch>1\
        and 'sgd' in params['optimizer'] else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params['bsize']):
        # prepare batch
        # print stidx
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params['bsize']], params['word_emb_dim'])
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params['bsize']], params['word_emb_dim'])

        tgt_batch = Variable(
            torch.LongTensor(target[stidx:stidx + params['bsize']]))
        if args.cuda:
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(
                torch.LongTensor(target[stidx:stidx + params['bsize']])).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)

        k = s1_batch.size(1)  # actual batch size

        # model forward
        # print(s1_batch.data.shape,s1_len)
        # print(s1_len,s2_len)
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        # print('output.size')
        # print(output.size())
        # print output.data
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params['bsize']])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        # the  Use tensor.item() to convert a 0-dim tensor to a Python number
        # all_costs.append(loss.item())
        # print len(all_costs)
        words_count += (
            s1_batch.nelement() + s2_batch.nelement()) / params['word_emb_dim']

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm()**2
        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            shrink_factor = max_norm / total_norm
        current_lr = optimizer.param_groups[0][
            'lr']  # current lr (no external "lr", for adam)
        optimizer.param_groups[0][
            'lr'] = current_lr * shrink_factor  # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append(
                '{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.
                format(stidx, round(np.mean(all_costs), 2),
                       int(
                           len(all_costs) * params['bsize'] /
                           (time.time() - last_time)),
                       int(words_count * 1.0 / (time.time() - last_time)),
                       round(100. * correct / (stidx + k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct / len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'.format(
        epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop
    global n_earlystopping

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    # s1 = X_train_passage if eval_type == 'valid' else X_val_passage
    # s2 = X_train_query if eval_type == 'valid' else X_val_query
    # target = y_train if eval_type == 'valid' else y_val
    s1=X_val_passage
    s2=X_val_query
    target=y_val

    for i in range(0, len(s1), params['bsize']):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params['bsize']],params['word_emb_dim'])
        s2_batch, s2_len = get_batch(s2[i:i + params['bsize']],params['word_emb_dim'])

        tgt_batch = Variable(torch.LongTensor(target[i:i + params['bsize']]))
        if args.cuda:
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i:i + params['bsize']])).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)


        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)

            state = nli_net.state_dict()
            for key in state: state[key] = state[key].clone().cpu()
            torch.save(state,os.path.join(outputdir, outputmodelname))
            torch.save(nli_net.state_dict(),
                    os.path.join(outputdir, outputmodelnamecuda))
            n_earlystopping=0
            val_acc_best = eval_acc
        else:
            # if 'sgd' in params['optimizer']:
            #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0][
            #         'lr'] / lrshrink
            #     print('Shrinking lr by : {0}. New lr = {1}'.format(
            #         lrshrink, optimizer.param_groups[0]['lr']))
            #     if optimizer.param_groups[0]['lr'] < minlr:
            #         stop_training = True
            n_earlystopping += 1
            print 'adam early stopping (at 7nd decrease in accuracy) and now is {}'.format(n_earlystopping)
            if n_earlystopping>50:
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


def error_classify(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop
    global n_earlystopping

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = X_train_passage if eval_type == 'valid' else X_val_passage
    s2 = X_train_query if eval_type == 'valid' else X_val_query
    target = y_train if eval_type == 'valid' else y_val
    query_id = query_id_train if eval_type=='valid' else query_id_val

    f_error=get_data_path()+eval_type+'error.csv'
    f_right=get_data_path()+eval_type+'right.csv'
    label_cnt = {}
    with open(f_error,'wb') as wf1,open(f_right,'wb') as wf2:
        for i in range(0, len(s1), params['bsize']):
            # prepare batch
            s1_batch, s1_len = get_batch(s1[i:i + params['bsize']],params['word_emb_dim'])
            s2_batch, s2_len = get_batch(s2[i:i + params['bsize']],params['word_emb_dim'])

            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
            tgt_batch = Variable(torch.LongTensor(target[i:i + params['bsize']]))
            query_id_batch = query_id[i:i + params['bsize']]



            # model forward
            output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
            pred = output.data.max(1)[1]

            for q,t,p,_s1,_s2 in zip(query_id_batch,tgt_batch,pred,s1[i:i + params['bsize']],s2[i:i + params['bsize']]):
                if label_cnt.has_key(int(t)):
                    label_cnt[int(t)]+=1
                else:
                    label_cnt[int(t)]=1

                if int(t)==int(p):
                    wf2.write('{}\t{}\t{}\t{}\t{}\n'.format(q,t,p,' '.join(_s1),' '.join(_s2)))
                else:
                    wf1.write('{}\t{}\t{}\t{}\t{}\n'.format(q,t,p,' '.join(_s1),' '.join(_s2)))


            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()


    print(label_cnt)
    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

"""
Train model on Natural Language Inference task
"""

# if exit model load it
# if os.path.exists(os.path.join(outputdir, outputmodelname)):
#     nli_net.load_state_dict(torch.load(os.path.join(outputdir, outputmodelname)))

epoch = 1
# n_epochs =2
while not stop_training and epoch <= n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
nli_net.load_state_dict(torch.load(os.path.join(outputdir, outputmodelname)))

# print('\nTEST : Epoch {0}'.format(epoch))
# evaluate(1e6, 'valid', True)
# evaluate(0, 'test', True)

# print ('\n error classify')
# error_classify(1e6,'valid',True)

# Save encoder instead of full model
torch.save(nli_net.encoder.state_dict(),
           os.path.join(outputdir, outputmodelname + '.encoder.pkl'))
