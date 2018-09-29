# -*- coding:utf-8 -*-
"""
  transformatted sentence to fixed size vec
"""
import gensim.models.word2vec as w2v
from gensim.test.utils import datapath
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import time

model_file_name = './corpus/w2v.model.bin'



#模型训练，生成词向量
def train_word2vec():
    # sentences = w2v.LineSentence('./data/all_corpus.csv')
    sentences = w2v.LineSentence('./data/all_corpus2.csv')
    # sentences = w2v.LineSentence('./data/oqmrc_validationset_corpus.csv')
    model = w2v.Word2Vec(sentences, size=128, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    word_vectors.save_word2vec_format(model_file_name, binary=True)


def load_word2vec():
    train_w2v = KeyedVectors.load_word2vec_format(model_file_name, binary=True)
    return train_w2v


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


## 训练word2vec
train_word2vec()

test_seq = [u'是', u'不是']
train_w2v = load_word2vec()
print('train_w2v length is {}'.format(len(train_w2v.vocab)))
print('train_w2v vector size is {}'.format(train_w2v.vector_size))
print(word2vec_seq(test_seq, train_w2v))
sys.exit()



import numpy as np
import torch
from nnmodel import NLINet
"""
SEED
"""
np.random.seed(2048)
torch.manual_seed(2048)
"""
train
"""
# parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
params = {
    'n_words': len(train_w2v.vocab),
    'word_emb_dim': train_w2v.vector_size,
    'enc_lstm_dim': 512,
    'n_enc_layers': 1,
    'dpout_model': 0.,
    'dpout_fc': 0.,
    'fc_dim': 128,
    'bsize': 64,
    'n_classes': 3,
    'pool_type': 'max',
    'nonlinear_fc': 0,
    'encoder_type': 'InferSentV1',
    'use_cuda': False,
}
nli_net = NLINet(params)
print(nli_net)

# loss
weight = torch.FloatTensor(params['n_classes']).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

from torch import optim
# optimizer
optimizer =optim.Adam(nli_net.parameters())

# cuda by default
# nli_net.cuda()
# loss_fn.cuda()

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
max_norm=5.
lr = None

from data import load_classifytraindata,get_data_path,get_batch
from torch.autograd import Variable

trainf = get_data_path() + 'oqmrc_trainingset_classify.csv'
valf = get_data_path() + 'oqmrc_validationset_classify.csv'
X_train_passage, X_val_passage, X_train_query, X_val_query, y_train, y_val = load_classifytraindata(trainf,valf)



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
    print(permutation)

    s1 = X_train_passage[permutation]
    s2 = X_train_query[permutation]
    target = y_train[permutation]

    print 'beee'

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params['bsize']):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params['bsize']],
                                     train_w2v, params['word_emb_dim'])
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params['bsize']],
                                     train_w2v, params['word_emb_dim'])
        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params['bsize']]))
        k = s1_batch.size(1)  # actual batch size

        # model forward
        # print(s1_batch)
        # print(s1_len,s2_len)
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params['bsize']])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
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

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], train_w2v,
                                     params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], train_w2v,
                                     params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(
            s2_batch.cuda())
        tgt_batch = Variable(
            torch.LongTensor(target[i:i + params.batch_size])).cuda()

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

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(),
                       os.path.join(params.outputdir, params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0][
                    'lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'.format(
                    params.lrshrink, optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1
n_epochs =10
while not stop_training and epoch <= n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
nli_net.load_state_dict(
    torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

# Save encoder instead of full model
torch.save(nli_net.encoder.state_dict(),
           os.path.join(params.outputdir,
                        params.outputmodelname + '.encoder.pkl'))

# from nnmodel import InferSent
# #TODO test 300 dim
# # word_emb_dim=300
# params_model = {'bsize': 64, 'word_emb_dim': 64, 'enc_lstm_dim': 2048,
#                 'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
