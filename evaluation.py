from itertools import chain
import argparse
import pandas as pd
from nltk.tag import pos_tag
from stanfordcorenlp import StanfordCoreNLP
import spacy
from flair.nn import Classifier
from flair.data import Sentence

from corpus import Corpus


def nltk_pos_tag(identifiers, tags):
    print('nltk')
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    out_tags = []
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        result = pos_tag(identifier)
        result_tags = [tag for _, tag in result]

        out_tags.append(result_tags)

        if tag == result_tags:
            correct_ids += 1

        for j, (golden, out) in enumerate(zip(tag, result_tags)):
            if golden == out:
                correct_tokens += 1
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def stanford_pos_tag(identifiers, tags, path2stanford):
    nlp = StanfordCoreNLP(path2stanford)
    # print(nlp.pos_tag('Get Name'))
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    out_tags = []
    wrong_tokenization = 0
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        stanford_input = " ".join(identifier)
        result = nlp.pos_tag(stanford_input)
        result_tags = [tag for _, tag in result]
        tokens = [token for token, _ in result]

        if tag == result_tags:
            correct_ids += 1

        if len(result_tags) == len(identifier):
            for j, (golden, out) in enumerate(zip(tag, result_tags)):
                if golden == out:
                    correct_tokens += 1
        elif len(result_tags) > len(tag):
            for k, token in enumerate(identifier):
                if token == tokens[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                else:
                    for l in range(k+1, len(tokens)):
                        temp_token = ''.join(tokens[k:l+1])
                        if temp_token == token:
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
                        if ('+'+temp_token) == token:
                            # + and wrong number tokenization
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, '+'+temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
                # except IndexError as e:
                #     print(k, identifier[k], tokens, len(tag), tag)
        elif len(result_tags) < len(tag):
            # nl tel number may be wrong tokenized "1 201 123" will be processed as one token
            for k, token in enumerate(tokens):
                if token == identifier[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                elif identifier[k] == '+':
                    # + is missing in this situation
                    tokens.insert(k, '+')
                    result_tags.insert(k, '<UNK>')
                    wrong_tokenization += 1
                else:
                    for l in range(k+1, len(identifier)):
                        temp_token = ' '.join(identifier[k:l+1])
                        if temp_token == token:
                            del tokens[k]
                            del result_tags[k]
                            for m in range(k, l+1):
                                tokens.insert(m, identifier[m])
                                result_tags.insert(m, '<UNK>')
                                wrong_tokenization += 1
                            break
        if len(result_tags) != len(identifier):
            print('Wrong sequence: ', identifier, tag)
            print(tokens, result_tags)

        out_tags.append(result_tags)


    print('Stanford CoreNLP')
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def spacy_pos_tag(identifiers, tags):
    nlp = spacy.load('en_core_web_sm')
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    wrong_tokenization = 0
    out_tags = []
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        spacy_input = " ".join(identifier)
        result = nlp(spacy_input)
        result_tags = [token.tag_ for token in result]
        tokens = [token.text for token in result]

        if tag == result_tags:
            correct_ids += 1

        if len(result_tags) == len(identifier):
            for j, (golden, out) in enumerate(zip(tag, result_tags)):
                if golden == out:
                    correct_tokens += 1
        else:
            # id will be wrong tokenized as i d
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


        out_tags.append(result_tags)


        # if len(result_tags) != len(identifier):
        #     print(result_tags, identifier, spacy_input, [token for token in result])
        # if tag == result_tags:
        #     correct_ids += 1
        # for j, (golden, out) in enumerate(zip(tag, result_tags)):
        #     if golden == out:
        #         correct_tokens += 1
    print('Spacy')
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def flair_pos_tag(identifiers, tags):
    tagger = Classifier.load('pos')
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


        out_tags.append(result_tags)
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


def list2str(lists):
    new_lists = []
    for l in lists:
        new_lists.append(' '.join(l))
    return new_lists



if __name__ == '__main__':
    # nltk.download('averaged_perceptron_tagger')
    parser = argparse.ArgumentParser(description='Evaluation on four taggers')
    # mode id or nl
    parser.add_argument('-m', '--mode', help='Choose id or nl to process.')
    args = parser.parse_args()
    eva_mode = args.mode
    print(eva_mode)
    if eva_mode == 'id':
        # print(eva_mode)
        corpus = Corpus('./dataset')
        nltk_out = nltk_pos_tag(corpus.test_input, corpus.test_tags)
        stanford_out = stanford_pos_tag(corpus.test_input, corpus.test_tags, path2stanford='../corenlp')
        spacy_out = spacy_pos_tag(corpus.test_input, corpus.test_tags)
        flair_out = flair_pos_tag(corpus.test_input, corpus.test_tags)
        # flair_pos_tag()
        data = {'SEQUENCE': list2str(corpus.test_input),
                'POS': list2str(corpus.test_tags),
                'NLTK': list2str(nltk_out),
                'STANFORD': list2str(stanford_out),
                'SPACY': list2str(spacy_out),
                'FLAIR': list2str(flair_out)}
        df = pd.DataFrame(data)
        df.to_csv('./evaluation_id.csv')
    elif eva_mode == 'nl':
        # print(eva_mode)
        data = pd.read_csv('./dataset/nl_data.csv')
        sentences = data['SEQUENCE'].values.tolist()
        poses = data['POS'].values.tolist()

        sentences = [sentence.split(' ') for sentence in sentences]
        poses = [pos.split(' ') for pos in poses]

        nltk_out = nltk_pos_tag(sentences, poses)
        stanford_out = stanford_pos_tag(sentences, poses, path2stanford='../corenlp')
        spacy_out = spacy_pos_tag(sentences, poses)
        flair_out = flair_pos_tag(sentences, poses)
        # flair_pos_tag()
        data = {'SEQUENCE': list2str(sentences),
                'POS': list2str(poses),
                'NLTK': list2str(nltk_out),
                'STANFORD': list2str(stanford_out),
                'SPACY': list2str(spacy_out),
                'FLAIR': list2str(flair_out)}
        df = pd.DataFrame(data)
        df.to_csv('./evaluation_nl.csv')