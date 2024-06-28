import os
import copy

from corpus import Corpus
import pandas as pd
from nltk import UnigramTagger, BigramTagger, CRFTagger, HiddenMarkovModelTagger

# from train_nltk import parse_data, train_hmm, train_crf, train_bigram, train_unigram

class Vocab:
    def __init__(self, corpus):
        self.w2i, self.i2w = self.make_vocab(corpus)
        self.oov_cases, self.oov_ids, self.oov_vocab, self.oov_id2file = self.get_test_oov(corpus)
        self.oov_w2i = {w:i for i, w in enumerate(self.oov_vocab)}
        assert len(self.w2i) == len(self.i2w)
        self.len = len(self.w2i)

    def __len__(self):
        return self.len

    def make_vocab(self, corpus):
        train_data = corpus.train_data
        dev_data = corpus.dev_data
        test_data = corpus.test_data
        data = [train_data, dev_data, test_data]
        word_list = []
        i2w = []
        w2i = dict()
        for i, data_part in enumerate(data):
            if data_part is None:
                continue
            if i==0:
                # process training data
                identifiers = data_part['identifier'].values.tolist()
                for identifier in identifiers:
                    word_list += identifier.split(' ')
                word_set = set(word_list)
                i2w = sorted(list(word_set))
                w2i = {w:j for j, w in enumerate(i2w)}
        return w2i, i2w

    def get_test_oov(self, corpus):
        test_data = corpus.test_data
        data = test_data['identifier'].values.tolist()
        label = test_data['pos_tag'].values.tolist()
        file = test_data['file'].values.tolist()

        ids = []
        oov_vocab = set()
        oov_id2file = dict()
        pair_data = []
        for i, id in enumerate(data):
            has_oov = False
            for w in id.split():
                if w not in self.i2w:
                    # print(w)
                    # is unlabeled word
                    # record word
                    oov_vocab.add(w)
                    # record id
                    if not has_oov:
                        # prevent redundancy
                        # record file
                        oov_id2file[id] = file[i]
                        ids.append(id.split(' '))
                        this_pair = []
                        for x, y in zip(data[i].split(' '), label[i].split(' ')):
                            pair = [tuple([w, l]) for w, l in zip(x.strip('\n').split(' '), y.strip('\n').split(' '))]
                            this_pair += pair
                        pair_data.append(this_pair)
                        # [[('apple', 'NN'), ], [('banana', 'NN')]]
                        has_oov = True
        # print(len(pair_data))
        return pair_data, ids, list(oov_vocab), oov_id2file


if __name__ == "__main__":
    # dir
    data_path = './data/'
    # data_path = './table/'
    # mode
    mode = ['method', 'class', 'args']
    corpus = Corpus(data_path, mode[0])
    vocab = Vocab(corpus)
    print(len(vocab.oov_cases))
    print(len(vocab.oov_vocab))
    print(vocab.i2w)

