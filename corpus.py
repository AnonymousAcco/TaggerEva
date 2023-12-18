import os
import pandas as pd


def get_data(data_path, file_name):
    path = os.path.join(data_path, file_name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


def get_words_and_tags(data):
    identifiers = data['SEQUENCE'].values.tolist()
    identifiers = [identifier.split(' ') for identifier in identifiers]
    tags = data['POS'].values.tolist()
    tags = [tag.split(' ') for tag in tags]
    return identifiers, tags


class Corpus:
    def __init__(self, data_path):
        self.train_data = get_data(data_path, 'train.csv')
        self.test_data = get_data(data_path, 'test.csv')
        self.dev_data = get_data(data_path, 'dev.csv')

        self.train_input, self.train_tags = get_words_and_tags(self.train_data)
        if self.dev_data is not None:
            self.dev_input, self.dev_tags = get_words_and_tags(self.dev_data)
        self.test_input, self.test_tags = get_words_and_tags(self.test_data)


if __name__ == "__main__":
    data_path = './dataset/'
    corpus = Corpus(data_path)
    print(corpus.test_tags)
