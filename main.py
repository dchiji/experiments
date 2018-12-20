
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

import argparse


# pos_batch, neg_batch: list of size (BATCH_SIZE, length for each questions)
def one_batch_train(model, pos_batch, neg_batch):
    pos_target = [[1.0]] * len(pos_batch)
    neg_target = [[0.0]] * len(neg_batch)
    target = pos_target + neg_target
    target = torch.FloatTensor(target).to(model.device)

    batch = pos_batch + neg_batch
    batch = model.padding(batch)
    batch = torch.LongTensor(batch).to(model.device)

    loss = nn.BCELoss()
    model.opt.zero_grad()
    pred = model(batch) # (batch_size, 1)
    out = loss(pred, target)
    out.backward()
    model.opt.step()

    return out.item()

def split_into_batches(model, data, opts):
    one_epoch = int(len(data['positive-seq']) / opts.batch_size)
    pos_batch_size = int(opts.batch_size / 2)
    neg_batch_size = int(opts.batch_size / 2)

    batches = []
    for i in range(one_epoch):
        pos_batch = data['positive-seq'][i * pos_batch_size:(i+1) * pos_batch_size]
        neg_batch = data['negative-seq'][i * neg_batch_size:(i+1) * neg_batch_size]

        pos_neg_batch = model.padding(pos_batch + neg_batch)
        pos_batch = pos_neg_batch[0:pos_batch_size]
        neg_batch = pos_neg_batch[pos_batch_size:]

        batches.append([pos_batch, neg_batch])
    return batches

def one_epoch_train(model, data, opts):
    total_loss = 0
    batches = split_into_batches(model, data, opts)
    for bat in batches:
        total_loss += one_batch_train(model, bat[0], bat[1])
    return total_loss

def one_batch_predict(model, batch, opts):
    batch = model.padding(batch)
    batch = torch.LongTensor(batch).to(model.device)
    pred = model(batch) # (batch_size, 1)
    return [1 if pred[i][0].item() > opts.threshold else 0 for i in range(len(batch))]

def one_epoch_eval(model, data, opts):
    total_acc = 0
    all_pred = []
    batches = []
    corrects = []
    seq_lis = [(d, 1) for d in data['positive-seq']] + [(d, 0) for d in data['negative-seq']]
    for i in range(int(len(seq_lis) / opts.batch_size)):
        sublist = seq_lis[i * opts.batch_size: (i+1) * opts.batch_size]
        bat = [p[0] for p in sublist]
        cor = [p[1] for p in sublist]
        batches.append(bat)
        corrects.append(cor)

    for i in range(len(batches)):
        pred = one_batch_predict(model, batches[i], opts)
        all_pred += pred
        total_acc += sum([1 if pred[j] == corrects[i][j] else 0 for j in range(len(pred))])
    return total_acc, all_pred

def start_train(model, data, data_test, opts):
    print('Start Training ...')

    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=opts.weight_decay)
    model.set_optimizer(opt)

    data['positive-seq'] = [model.text_to_idx_seq(text) for _, text in data['positive']]
    data['negative-seq'] = [model.text_to_idx_seq(text) for _, text in data['negative']]

    data_test['positive-seq'] = [model.text_to_idx_seq(text) for _, text in data_test['positive']]
    data_test['negative-seq'] = [model.text_to_idx_seq(text) for _, text in data_test['negative']]

    for epoch in range(opts.epoch):
        print('[Epoch ' + str(epoch) + ']')

        total_loss = one_epoch_train(model, data, opts)
        print('Train Loss: ' + str(total_loss / len(data['positive'])))

        print('Train Accuracy: ', end='')
        data_sub = {}
        data_sub['positive-seq'] = data['positive-seq'][0:int(len(data['positive'])*0.1)]
        data_sub['negative-seq'] = data['negative-seq'][0:int(len(data['positive'])*0.1)]
        acc, _ = one_epoch_eval(model, data_sub, opts)
        print(str(acc / (len(data_sub['positive-seq']) + len(data_sub['negative-seq']))))

        print('Test Accuracy: ', end='')
        acc, _ = one_epoch_eval(model, data_test, opts)
        print(str(acc / (len(data_test['positive-seq']) + len(data_test['negative-seq']))))

        if opts.save_path != '':
            torch.save(model, opts.save_path + '/' + opts.name + '_epoch_' + str(epoch) + '.pth')

def make_submission_csv(model, data_submission, opts):
    print('Make submission.csv ...')
    out = []
    out.append('qid,prediction\n')

    data_submission['positive-seq'] = [model.text_to_idx_seq(text) for _, text in data_submission['positive']]
    data_submission['negative-seq'] = [model.text_to_idx_seq(text) for _, text in data_submission['negative']]
    _, pred = one_epoch_eval(model, data_submission, opts)

    insincere_num = 0
    sincere_num = 0

    for [qid, text], p in zip(data_submission['positive'], pred):
        out.append('%s,%d\n' % (qid, 1 - p))
        insincere_num += 1 - p
        sincere_num += p
    with open('submission.csv', 'w') as f:
        f.writelines(out)
    print('SincereQ: %d \t InsincereQ: %d' % (sincere_num, insincere_num))

def main(DATA, DATA_TEST, DATA_SUBMISSION, opts):
    # Initialization
    idx2word = []
    word2idx = {}

    if opts.debug:
        DATA['positive'] = DATA['positive'][:500]
        DATA['negative'] = DATA['negative'][:500]
        DATA_TEST['positive'] = DATA_TEST['positive'][:500]
        DATA_TEST['negative'] = DATA_TEST['negative'][:500]
        opts.emb_dim = 50
        opts.epoch = 1

    EMB_PKL = 'embedding_' + str(opts.emb_dim) + '.pickle'
    if not os.path.exists(EMB_PKL):
        print('Load GloVe vector file ...')
        with open('glove.twitter.27B.'+str(opts.emb_dim)+'d.txt') as f:
            str_vecs = f.readlines()

        idx2tensor = []
        for i in range(len(str_vecs)):
            line = str_vecs[i].split(' ')
            idx2word.append(line[0])
            word2idx[line[0]] = i
            idx2tensor.append(torch.FloatTensor([[float(s) for s in line[1:]]]))
        idx2tensor.append(torch.zeros([1,opts.emb_dim]).float())
        word2idx['<padding>'] = len(idx2tensor) - 1
        init_weight = torch.cat(idx2tensor, dim=0)

        if not opts.save_disc_capacity:
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

    init_weight = init_weight.to(opts.device)


    # Start Training
    if opts.load_path == '' or not os.path.exists(opts.load_path) or opts.forced_train:
        if opts.model == 'classifier':
            model = Classifier(opts.emb_dim, init_weight, opts.device)
        elif opts.model == 'gru':
            model = GRUBase(opts.emb_dim, idx2word, word2idx, init_weight, opts.device, opts.gru_hidden)
        else:
            raise Exception
        model.to(opts.device)
        start_train(model, DATA, DATA_TEST, opts)
    else:
        model = torch.load(opts.load_path)
        model.to(opts.device)
        model.eval()
    make_submission_csv(model, DATA_SUBMISSION, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
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
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('data_train.pickle', 'rb') as f:
        DATA = pickle.load(f)
    with open('data_test.pickle', 'rb') as f:
        DATA_TEST = pickle.load(f)
    with open('data_submission.pickle', 'rb') as f:
        DATA_SUBMISSION = pickle.load(f)

    main(DATA, DATA_TEST, DATA_SUBMISSION, opts)
