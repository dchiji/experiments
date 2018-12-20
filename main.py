
# TODO
#   - Grid search for hyperparams with matplotlib
#   - Cross validation (because # of validation set < 1000)
#   - Ensembled model

import torch
import torch.nn as nn
import torch.optim as optim

from model import *

import os
import pickle
from IPython import embed
from collections import defaultdict


idx2word = []
word2idx = {}

with open('data_train.pickle', 'rb') as f:
    data = pickle.load(f)

with open('data_test.pickle', 'rb') as f:
    data_test = pickle.load(f)

with open('data_submission.pickle', 'rb') as f:
    data_submission = pickle.load(f)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--model', type=str, default='classifier')
parser.add_argument('--gru_hidden', type=int, default=90)
parser.add_argument('--save_disc_capacity', type=bool, default=False)
parser.add_argument('--forced-train', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0)
opts = parser.parse_args()

BATCH_SIZE = opts.batch
EMB_DIM = opts.emb_dim
EPOCH = opts.epoch
EMB_PKL = 'embedding_' + str(EMB_DIM) + '.pickle'

MODEL_TYPE = opts.model
MODEL_NAME = opts.name
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FORCED_TRAIN_FLAG = opts.forced_train
DEBUG_FLAG = opts.debug
SAVE_DISC_CAPACITY_FLAG = opts.save_disc_capacity
SAVE_PATH = opts.save_path
LOAD_PATH = opts.load_path
GRU_HIDDEN_DIM = opts.gru_hidden
THRESHOLD = opts.threshold
WEIGHT_DECAY = opts.weight_decay


if not os.path.exists(EMB_PKL):
    print('Load GloVe vector file ...')
    with open('glove.twitter.27B.'+str(EMB_DIM)+'d.txt') as f:
        str_vecs = f.readlines()

    idx2tensor = []
    for i in range(len(str_vecs)):
        line = str_vecs[i].split(' ')
        idx2word.append(line[0])
        word2idx[line[0]] = i
        idx2tensor.append(torch.FloatTensor([[float(s) for s in line[1:]]]))
    idx2tensor.append(torch.zeros([1,EMB_DIM]).float())
    word2idx['<padding>'] = len(idx2tensor) - 1
    init_weight = torch.cat(idx2tensor, dim=0)

    if not SAVE_DISC_CAPACITY_FLAG:
        with open(EMB_PKL, 'wb') as f:
            dic = {'idx2word': idx2word, 'word2idx': word2idx, 'init_weight': init_weight}
            pickle.dump(dic, f)
else:
    print('Load ' + EMB_PKL + ' ...')
    with open(EMB_PKL, 'rb') as f:
        dic = pickle.load(f)
        idx2word = dic['idx2word']
        word2idx = dic['word2idx']
        init_weight = dic['init_weight']

init_weight = init_weight.to(DEVICE)


# pos_batch, neg_batch: list of size (BATCH_SIZE, length for each questions)
def one_batch_train(model, pos_batch, neg_batch):
    pos_target = [[1.0]] * len(pos_batch)
    neg_target = [[0.0]] * len(neg_batch)
    target = pos_target + neg_target
    target = torch.FloatTensor(target).to(DEVICE)

    batch = pos_batch + neg_batch
    batch = model.padding(batch)
    batch = torch.LongTensor(batch).to(DEVICE)

    loss = nn.BCELoss()
    model.opt.zero_grad()
    pred = model(batch) # (batch_size, 1)
    out = loss(pred, target)
    out.backward()
    model.opt.step()

    return out.item()

def split_into_batches(model, data):
    one_epoch = int(len(data['positive-seq']) / BATCH_SIZE)
    pos_batch_size = int(BATCH_SIZE / 2)
    neg_batch_size = int(BATCH_SIZE / 2)

    batches = []
    for i in range(one_epoch):
        pos_batch = data['positive-seq'][i * pos_batch_size:(i+1) * pos_batch_size]
        neg_batch = data['negative-seq'][i * neg_batch_size:(i+1) * neg_batch_size]

        pos_neg_batch = model.padding(pos_batch + neg_batch)
        pos_batch = pos_neg_batch[0:pos_batch_size]
        neg_batch = pos_neg_batch[pos_batch_size:]

        batches.append([pos_batch, neg_batch])
    return batches

def one_epoch_train(model, data):
    total_loss = 0
    batches = split_into_batches(model, data)
    for bat in batches:
        total_loss += one_batch_train(model, bat[0], bat[1])
    return total_loss

def one_batch_predict(model, batch):
    batch = model.padding(batch)
    batch = torch.LongTensor(batch).to(DEVICE)
    pred = model(batch) # (batch_size, 1)
    return [1 if pred[i][0].item() > THRESHOLD else 0 for i in range(len(batch))]

def one_epoch_eval(model, data):
    total_acc = 0
    all_pred = []
    batches = []
    corrects = []
    seq_lis = [(d, 1) for d in data['positive-seq']] + [(d, 0) for d in data['negative-seq']]
    for i in range(int(len(seq_lis) / BATCH_SIZE)):
        sublist = seq_lis[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
        bat = [p[0] for p in sublist]
        cor = [p[1] for p in sublist]
        batches.append(bat)
        corrects.append(cor)

    for i in range(len(batches)):
        pred = one_batch_predict(model, batches[i])
        all_pred += pred
        total_acc += sum([1 if pred[j] == corrects[i][j] else 0 for j in range(len(pred))])
    return total_acc, all_pred

def start_train(model):
    print('Start Training ...')

    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
    model.set_optimizer(opt)

    data['positive-seq'] = [model.text_to_idx_seq(text) for _, text in data['positive']]
    data['negative-seq'] = [model.text_to_idx_seq(text) for _, text in data['negative']]

    data_test['positive-seq'] = [model.text_to_idx_seq(text) for _, text in data_test['positive']]
    data_test['negative-seq'] = [model.text_to_idx_seq(text) for _, text in data_test['negative']]

    for epoch in range(EPOCH):
        print('[Epoch ' + str(epoch) + ']')

        total_loss = one_epoch_train(model, data)
        print('Train Loss: ' + str(total_loss / len(data['positive'])))

        print('Train Accuracy: ', end='')
        data_sub = {}
        data_sub['positive-seq'] = data['positive-seq'][0:int(len(data['positive'])*0.1)]
        data_sub['negative-seq'] = data['negative-seq'][0:int(len(data['positive'])*0.1)]
        acc, _ = one_epoch_eval(model, data_sub)
        print(str(acc / (len(data_sub['positive-seq']) + len(data_sub['negative-seq']))))

        print('Test Accuracy: ', end='')
        acc, _ = one_epoch_eval(model, data_test)
        print(str(acc / (len(data_test['positive-seq']) + len(data_test['negative-seq']))))

        if SAVE_PATH != '':
            torch.save(model, SAVE_PATH + '/' + MODEL_NAME + '_epoch_' + str(epoch) + '.pth')

def make_submission_csv(model):
    print('Make submission.csv ...')
    out = []
    out.append('qid,prediction\n')

    data_submission['positive-seq'] = [model.text_to_idx_seq(text) for _, text in data_submission['positive']]
    data_submission['negative-seq'] = [model.text_to_idx_seq(text) for _, text in data_submission['negative']]
    _, pred = one_epoch_eval(model, data_submission)

    insincere_num = 0
    sincere_num = 0

    for [qid, text], p in zip(data_submission['positive'], pred):
        out.append('%s,%d\n' % (qid, 1 - p))
        insincere_num += 1 - p
        sincere_num += p
    with open('submission.csv', 'w') as f:
        f.writelines(out)
    print('SincereQ: %d \t InsincereQ: %d' % (sincere_num, insincere_num))

if __name__ == '__main__':
    if DEBUG_FLAG:
        embed()
    if LOAD_PATH == '' or not os.path.exists(LOAD_PATH) or FORCED_TRAIN_FLAG:
        if MODEL_TYPE == 'classifier':
            model = Classifier(EMB_DIM, init_weight, DEVICE)
        elif MODEL_TYPE == 'gru':
            model = GRUBase(EMB_DIM, idx2word, word2idx, init_weight, DEVICE, GRU_HIDDEN_DIM)
        else:
            raise Exception
        model.to(DEVICE)
        start_train(model)
    else:
        model = torch.load(LOAD_PATH)
        model.to(DEVICE)
        model.eval()
    make_submission_csv(model)
