from itertools import chain
import argparse
from nltk.tag import PerceptronTagger
from nltk.corpus import wordnet as wn
import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example
import pandas as pd
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.nn import Classifier
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus


from corpus import Corpus
from evaluation import list2str


def train_nltk(train_ids, train_poses):
    print('nltk:')
    print("perceptron model")
    # brown_tagged_sents = brown.tagged_sents()
    # print(brown_tagged_sents)
    # [[('the', ''),
    training_data = []
    for id, pos in zip(train_ids, train_poses):
        pairs = [(word, tag) for word, tag in zip(id, pos)]
        training_data.append(pairs)

    # print(brown_tagger.evaluate(test_data))
    trained_tagger = PerceptronTagger()
    trained_tagger.train(training_data, './model/nltk/retrain.per.tagger.pickle')

    print(trained_tagger.accuracy(training_data))
    return trained_tagger

def eva_nltk(ids, poses, model_path='./model/nltk', model_name='retrain.per.tagger'):
    print('eva-mode: nltk:')
    print("perceptron model")
    # brown_tagged_sents = brown.tagged_sents()
    # print(brown_tagged_sents)
    # [[('the', ''),
    testing_data = []
    for id, pos in zip(ids, poses):
        pairs = [(word, tag) for word, tag in zip(id, pos)]
        testing_data.append(pairs)

    # print(brown_tagger.evaluate(test_data))
    trained_tagger = PerceptronTagger()
    trained_tagger.load('./'+model_path+'/'+model_name+'.pickle')
    # trained_tagger.train(training_data, 'retrain.per.tagger')

    print('Acc: ')
    print(trained_tagger.accuracy(testing_data))

    out_tag = []
    for id in ids:
        pos = trained_tagger.tag(id)
        tags = [tag for _, tag in pos]
        out_tag.append(" ".join(tags))
    return out_tag

    # return trained_tagger

def read_stanford(path='stanford_out.txt'):
    out = []
    with open(path, 'r') as file:
        stanford_out = file.readlines()
        for line in stanford_out:
            pairs = line.strip().split(' ')
            tags = [pair.split('/')[-1] for pair in pairs]
            out.append(" ".join(tags))
    return out

def create_stanford_data(ids, poses, out_name='./dataset/stanford_format/stanford_train.txt'):
    with open(out_name, 'w') as file:
        for id, tags in zip(ids, poses):
            line = []
            for word, tag in zip(id, tags):
                line.append(word+'/'+tag)
            out_line = ' '.join(line) + '\n'
            file.write(out_line)

def preprocess2spacy(ids, poses, data_type='train'):
    nlp = spacy.blank('en')
    data = []
    db = DocBin()
    for id, pos in zip(ids, poses):
        words = id
        tags = pos
        nlp(" ".join(id))
        doc = Doc(nlp.vocab, words=words, tags=tags)
        # for word in doc:
        #     print(word)
        # doc.tags = tags
        example = Example.from_dict(doc, {'words':words, 'tags':tags})

        db.add(doc)
    db.to_disk('./dataset/model/spacy_format'+data_type+'.spacy')



def test_spacy(ids, poses):
    nlp = spacy.load('./model/spacy/model-best')
    out_list = []
    for id, pos in zip(ids, poses):
        doc = nlp(" ".join(id))
        tags = []
        for word in doc:
            tags.append(word.tag_)
        out_list.append(' '.join(tags))
    return out_list


def train_flair(train_ids, train_poses):
    # data
    training_data = []
    with open('./dataset/flair_format/flair_train.txt', 'w') as file:
        for id, pos in zip(train_ids, train_poses):
            for word, tag in zip(id, pos):
                line = word + ' ' + tag + '\n'
                file.write(line)
            file.write('\n')

    columns = {0: 'text', 1: 'pos'}
    data_folder = './dataset/flair_format'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='flair_train.txt',
                                  )
    # test_file='test.txt',
    # dev_file='dev.txt'

    label_type = 'pos'
    embedding_types = [
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    # default_tagger = Classifier.load('pos')
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)
    trainer = ModelTrainer(tagger, corpus)

    trainer.train('./model/flair',
                  learning_rate=0.1,
                  mini_batch_size=32)

    sentence = Sentence('get name')
    print(tagger.predict(sentence))
    return tagger

def eva_flair(identifiers, tags):
    tagger = Classifier.load('./model/flair/best-model.pt')
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    wrong_tokenization = 0
    out_tags = []
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        flair_input = " ".join(identifier)
        # print(flair_input)
        sentence = Sentence(flair_input)
        tagger.predict(sentence)
        # print(sentence.get_labels()[0].value)
        # break
        result_tags = [token.value for token in sentence.get_labels()]
        tokens = [token.text for token in sentence.tokens]

        # #wordnet
        # for token in tokens:
        #
        #     w_description = wn.synsets(token)
        #     # print(w_description)
        #     if len(w_description) == 0:
        #         print("fw?"+token)


        if tag == result_tags:
            correct_ids += 1

        if len(result_tags) == len(identifier):
            for j, (golden, out) in enumerate(zip(tag, result_tags)):
                if golden == out:
                    correct_tokens += 1
        else:
            for k, token in enumerate(identifier):
                if token == tokens[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                else:
                    for l in range(k+1, len(tokens)):
                        temp_token = ''.join(tokens[k:l+1])
                        # print(temp_token)
                        if temp_token == token:
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
        if len(result_tags) != len(identifier):
            print('Wrong sequence: ', identifier, tag)
            print(tokens, result_tags)


        out_tags.append(" ".join(result_tags))
        # if len(result_tags) != len(identifier):
        #     print(result_tags, identifier, flair_input, tag)
        #
        # if tag == result_tags:
        #     correct_ids += 1
        # for j, (golden, out) in enumerate(zip(tag, result_tags)):
        #     if golden == out:
        #         correct_tokens += 1
    print('Flair')
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain taggers. -m ')
    parser.add_argument('-m', '--mode', help='Choose the mode: preprocess data or retrain. pre/train/eva')
    args = parser.parse_args()
    corpus = Corpus('./dataset')
    if args.mode == 'pre':
        create_stanford_data(corpus.train_input, corpus.train_tags)
        create_stanford_data(corpus.test_input, corpus.test_tags, 'stanford_test.txt')
        create_stanford_data(corpus.dev_input, corpus.dev_tags, 'stanford_dev.txt')

        preprocess2spacy(corpus.train_input, corpus.train_tags)
        preprocess2spacy(corpus.test_input, corpus.test_tags, 'test')
        preprocess2spacy(corpus.dev_input, corpus.dev_tags, 'dev')
    elif args.mode == 'train':
        nltk_tagger = train_nltk(corpus.train_input, corpus.train_tags)
        eva_nltk(corpus.dev_input, corpus.dev_tags)
        nltk_out = eva_nltk(corpus.test_input, corpus.test_tags)
        print(nltk_out)
        stanford_out = read_stanford()
        print(stanford_out)
        train_flair(corpus.train_input, corpus.train_tags)
        flair_out = eva_flair(corpus.test_input, corpus.test_tags)
        print(flair_out)
        spacy_out = test_spacy(corpus.test_input, corpus.test_tags)
        data = {'sentence': list2str(corpus.test_input),
                'pos': list2str(corpus.test_tags),
                'nltk': nltk_out,
                'stanford': stanford_out,
                'spacy': spacy_out,
                'flair': flair_out}
        df = pd.DataFrame(data)
        df.to_csv('./retrain_out.csv')
    elif args.mode == 'eva':
        nltk_out = eva_nltk(corpus.test_input, corpus.test_tags)
        stanford_out = read_stanford()
        flair_out = eva_flair(corpus.test_input, corpus.test_tags)
        spacy_out = test_spacy(corpus.test_input, corpus.test_tags)