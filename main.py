
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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--model', type=str, default='classifier')
parser.add_argument('--save_disc_capacity', type=bool, default=False)
parser.add_argument('--forced-train', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)
opts = parser.parse_args()

BATCH_SIZE = opts.batch
EMB_DIM = opts.emb_dim
EPOCH = opts.epoch
EMB_PKL = 'embedding_' + str(EMB_DIM) + '.pickle'

MODEL_TYPE = opts.model
MODEL_PATH = 'pth_' + opts.name
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FORCED_TRAIN_FLAG = opts.forced_train
DEBUG_FLAG = opts.debug
SAVE_DISC_CAPACITY_FLAG = opts.save_disc_capacity


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


def split_into_batches(data):
    one_epoch = int(len(data['positive']) / BATCH_SIZE)
    pos_batch_size = int(BATCH_SIZE / 2)
    neg_batch_size = int(BATCH_SIZE / 2)

    batches = []
    for i in range(one_epoch):
        pos_batch = []
        for _, text in data['positive'][i * pos_batch_size:(i+1) * pos_batch_size]:
            seq = []
            words = text.split(' ')
            for word in words:
                if word in word2idx:
                    seq.append(word2idx[word])
                else:
                    seq.append(word2idx['<unknown>'])
            pos_batch.append(seq)
        neg_batch = []
        for _, text in data['negative'][i * neg_batch_size:(i+1) * neg_batch_size]:
            seq = []
            words = text.split(' ')
            for word in words:
                if word in word2idx:
                    seq.append(word2idx[word])
                else:
                    seq.append(word2idx['<unknown>'])
            neg_batch.append(seq)

        # Padding
        max_len = max([len(seq) for seq in pos_batch + neg_batch])
        pos_batch = [seq + [word2idx['<padding>']] * (max_len - len(seq)) for seq in pos_batch]
        neg_batch = [seq + [word2idx['<padding>']] * (max_len - len(seq)) for seq in neg_batch]

        batches.append([pos_batch, neg_batch])
    return batches

# pos_batch, neg_batch: list of size (BATCH_SIZE, length for each questions)
def one_batch_train(model, pos_batch, neg_batch):
    max_len = max([len(text) for text in pos_batch + neg_batch])

    pos_target = [[1.0]] * len(pos_batch)
    neg_target = [[0.0]] * len(neg_batch)
    target = pos_target + neg_target
    target = torch.FloatTensor(target).to(DEVICE)

    # Padding
    batch = pos_batch + neg_batch
    batch = [idx_lis + [model.padding_idx] * (max_len - len(idx_lis)) for idx_lis in batch]
    batch = torch.LongTensor(batch).to(DEVICE)

    loss = nn.BCELoss()
    model.opt.zero_grad()
    pred = model(batch) # (batch_size, 1)
    out = loss(pred, target)
    out.backward()
    model.opt.step()

    return out.item()

def one_epoch_train(model, data):
    total_loss = 0
    batches = split_into_batches(data)
    for bat in batches:
        total_loss += one_batch_train(model, bat[0], bat[1])
    return total_loss

def one_batch_predict(model, batch):
    max_len = max([len(text) for text in batch])
    # Padding
    batch = [idx_lis + [model.padding_idx] * (max_len - len(idx_lis)) for idx_lis in batch]
    batch = torch.LongTensor(batch).to(DEVICE)
    pred = model(batch) # (batch_size, 1)
    return [1 if pred[i][0].item() > 0.5 else 0 for i in range(len(batch))]

def one_epoch_eval(model, data):
    total_acc = 0
    all_pred = []
    batches = split_into_batches(data)

    for bat in batches:
        pred = one_batch_predict(model, bat[0] + bat[1])
        all_pred += pred

        correct = ([1] * len(bat[0])) + ([0] * len(bat[1]))
        total_acc += sum([1 if pred[i] == correct[i] else 0 for i in range(len(pred))])
    return total_acc, all_pred

def start_train(model):
    print('Start Training ...')

    opt = optim.Adam(model.parameters(), lr=0.001)
    model.set_optimizer(opt)
    for epoch in range(EPOCH):
        print('[Epoch ' + str(epoch) + ']')

        total_loss = one_epoch_train(model, data)
        print('Train Loss: ' + str(total_loss / len(data['positive'])))

        print('Train Accuracy: ', end='')
        data_sub = {}
        data_sub['positive'] = data['positive'][0:int(len(data['positive'])*0.1)]
        data_sub['negative'] = data['negative'][0:int(len(data['positive'])*0.1)]
        acc, _ = one_epoch_eval(model, data_sub)
        print(str(acc / (len(data_sub['positive'] * 2))))

        print('Test Accuracy: ', end='')
        acc, _ = one_epoch_eval(model, data_test)
        print(str(acc / (len(data_test['positive'] * 2))))

if __name__ == '__main__':
    if DEBUG_FLAG:
        embed()
    if not os.path.exists(MODEL_PATH) or FORCED_TRAIN_FLAG:
        if MODEL_TYPE == 'classifier':
            model = Classifier(EMB_DIM, init_weight, DEVICE)
        if MODEL_TYPE == 'gru':
            model = GRUBase(EMB_DIM, init_weight, DEVICE)
        else:
            raise Exception
        model.to(DEVICE)
        start_train(model)
        torch.save(model, MODEL_PATH)
    else:
        model = torch.load(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
