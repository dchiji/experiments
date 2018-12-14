
from IPython import embed
import pickle
import csv
import re

pat1 = re.compile('([\\/\\(\\)\\,\\.\\?])[\\/\\(\\)\\,\\.\\?]*')   # (hoge,,fu/ba) => ( hoge , fu / ba )
pat2 = re.compile('(it|he|she)\'s ')    # it's hoge => it is hoge
pat3 = re.compile('\'m ')   # i'm => i am 
pat4 = re.compile('\'re ')   # they're => they are
pat5 = re.compile('(is|am|are|was|were|do|does|did|mayn|have|has|could|would)n\'t ')
pat6 = re.compile('(can\'t|cannot) ')
pat7 = re.compile('won\'t ')

pat8 = re.compile('\'s ')    # Mike's hoge => Mike s hoge

pat_space = re.compile('\s+')    # hoge  fuga => hoge fuga

def smoothing(text):
    if text[0] == text[-1] == '"':
        text = text[1:-1]
    text = text.lower()

    text = pat1.sub(' \\1 ', text)
    text = pat2.sub('\\1 is ', text)
    text = pat3.sub(' am ', text)
    text = pat4.sub(' are ', text)
    text = pat5.sub('\\1 not ', text)
    text = pat6.sub('can not ', text)
    text = pat7.sub('will not ', text)

    text = pat8.sub(' s ', text)

    text = pat_space.sub(' ', text)
    text = text.strip()

    return text

raw_csv = 'train.csv'
def make_pickle(rate=0.9):
    pos = []
    neg = []

    with open(raw_csv, 'r') as f:
        split_lines = csv.reader(f, delimiter=',')
        next(split_lines)

        for l in split_lines:
            qid, text, target = l
            text = smoothing(text)

            if target == '0':
                pos.append([qid, text])
            else:
                neg.append([qid, text])

    total_size = min(len(pos), len(neg))
    pos = pos[0:total_size]
    neg = neg[0:total_size]

    pos_train = pos[0:int(total_size * rate)]
    neg_train = neg[0:int(total_size * rate)]
    pos_test = pos[int(total_size * rate):]
    neg_test = neg[int(total_size * rate):]

    with open('data_train.pickle', 'wb') as f:
        dic = {'positive': pos_train, 'negative': neg_train}
        pickle.dump(dic, f)
    with open('data_test.pickle', 'wb') as f:
        dic = {'positive': pos_test, 'negative': neg_test}
        pickle.dump(dic, f)

if __name__ == '__main__':
    make_pickle()
