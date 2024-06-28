import os

import pandas as pd
from config import Config
from corpus import Corpus

def create_et_data(config, data_path, mode):
    columns = ['ID', 'TYPE', 'DECLARATION', 'PROJECT', 'FILE']
    corpus = Corpus(data_path, mode=config.mode)
    # print(corpus.test_data.columns)
    length = len(corpus.test_tags)

    # id
    id_list = list(range(length))
    # type
    type = []
    if mode == 'class':
        type = ['class'] * length
    elif mode == 'args':
        type = corpus.test_data['type'].values.tolist()
    # declaration
    declaration = corpus.test_data['ori_id'].values.tolist()
    #project
    project = corpus.test_data['project'].values.tolist()
    # file
    file = corpus.test_data['file'].values.tolist()

    data = {'ID': id_list, 'TYPE': type, 'DECLARATION': declaration, 'PROJECT': project, 'FILE':file}
    df = pd.DataFrame(data)
    df.to_csv(f'./{mode}_et_input.csv')
    return df




if __name__ == '__main__':
    data_path = './data/ensemble_format/'
    config = Config()
    data = create_et_data(config, data_path, mode='method')
    print(data)