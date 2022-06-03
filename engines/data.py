# -*- coding: utf-8 -*-
# @Time : 2021/7/25 19:02 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : data.py 
# @Software: PyCharm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from engines.utils.io_function import read_file, read_category
from engines.utils.buffer import Block, Buffer
from transformers import BertTokenizer
DEFAULT_MODEL_NAME = 'bert-base-chinese'
CAPACITY = 512 # Working Memory


class DataManager:
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger
        self.batch_size = configs.batch_size
        self.train_file = configs.datasets_fold + '/cnews/' + configs.train_file
        self.tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        self.vocabs_dir = configs.vocabs_dir
        self.label2id_file = self.vocabs_dir + '/label2id'
        self.label2id, self.id2label = self.load_label()
        self.max_label_number = len(self.label2id)

        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/cnews/' + configs.dev_file
        else:
            self.dev_file = None

    def clean(self, data):
        data_clean = data.replace('“', '"').replace('”', '"').replace('……', '').replace('—', '').lower()
        return data

    def load_label(self):
        if not os.path.isfile(self.label2id_file):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_labels(self.train_file)

        self.logger.info('loading label vocab...')
        label2id, id2label = {}, {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label_id
        return label2id, id2label

    def build_labels(self, train_path):
        df_train = read_file(train_path, names=['label', 'content'], delimiter='_!_')
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        label2id = dict(zip(labels, range(1, len(labels))))
        id2label = dict(zip(range(0, len(labels)), labels))
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')

        return label2id, id2label

    def get_training_set(self):
        print(self.configs.train_file)
        df_train = read_file(self.train_file, names=['label', 'content'], delimiter='_!_')
        categories, category_to_id = read_category()
        cnt, batches = 0, []
        for i in tqdm(range(len(df_train.content))):
        # for i in range(2000):
            d, label_name = self.clean(df_train.content[i]), df_train.label[i]
            if len(d) < 512:
                continue
            # print(d)
            l = category_to_id[label_name]
            qbuf, cnt = Buffer.split_document_into_blocks([self.tokenizer.cls_token], tokenizer=self.tokenizer,
                                                          cnt=cnt, properties=[('label_name', label_name), ('label', l), ('_id', str(i)), ('blk_type', 0)])
            dbuf, cnt = Buffer.split_document_into_blocks(self.tokenizer.tokenize(d), tokenizer=self.tokenizer, cnt=cnt)
            batches.append((qbuf, dbuf))
        with open(os.path.join(self.configs.datasets_fold, 'cnews', f'cnews_train.pkl'), 'wb') as fout:
            pickle.dump(batches, fout)
        return batches

    def get_valid_set(self):
        print(self.configs.dev_file)
        df_test = read_file(self.dev_file, names=['label', 'content'], delimiter='_!_')
        categories, category_to_id = read_category()
        cnt, batches = 0, []
        # for i in tqdm(range(len(df_test.content))):
        for i in range(20):
            d, label_name = self.clean(df_test.content[i]), df_test.label[i]
            l = category_to_id[label_name]
            qbuf, cnt = Buffer.split_document_into_blocks([self.tokenizer.cls_token], tokenizer=self.tokenizer,
                                                          cnt=cnt, properties=[('label_name', label_name), ('label', l), ('_id', str(i)), ('blk_type', 0)])
            dbuf, cnt = Buffer.split_document_into_blocks(self.tokenizer.tokenize(d), tokenizer=self.tokenizer, cnt=cnt)
            print(dbuf)
            batches.append((qbuf, dbuf))
        with open(os.path.join(self.configs.datasets_fold, 'cnews', f'cnews_test.pkl'), 'wb') as fout:
            pickle.dump(batches, fout)
        return batches

    def prepare(self, df):
        self.logger.info('loading data...')
        df = df.dropna()
        x = []
        y = []
        tmp_x = []
        tmp_y = []
        for index, record in tqdm(df.iterrows()):
            content = record.content
            label = record.label
            tmp_x = self.tokenizer.encode(content)
            x.append(tmp_x)
            tmp_y = self.label2id[label]
            tmp_x = []
            tmp_y = []
        return np.array(x), np.array(y)

    def next_batch(self, dataset, start_index):
        buf_batch = self.get_bufs(dataset, start_index)
        inputs = np.zeros(shape=(4, len(buf_batch), CAPACITY), dtype=np.int32)
        for i, buf in enumerate(buf_batch):
            buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        for i, buf in enumerate(buf_batch):
            buf.export_relevance(out=inputs[3, i])
        x_batch = inputs[0]
        att_mask_batch = inputs[1]
        token_type_ids_batch = inputs[2]
        y_batch = inputs[3]
        return buf_batch, x_batch, y_batch, att_mask_batch, token_type_ids_batch

    def get_bufs(self, dataset, start_index):
        last_index = start_index + self.batch_size
        bufs_batch = dataset[start_index:min(last_index, len(dataset))]
        if last_index > len(dataset):
            left_size = last_index - len(dataset)
            for i in range(left_size):
                index = np.random.randint(len(dataset))
                bufs_batch.append(dataset[index])

        return bufs_batch


# root_dir1 = os.path.abspath(os.path.dirname(__file__))
# config_file = os.path.join(root_dir1, 'system.config')
# configs = Configure(config_file=config_file)
# logger = get_logger(configs.log_dir)
# dataManager = DataManager(configs, logger)
# dataManager.get_training_set()


# root_dir2 = os.path.abspath(os.path.dirname((os.path.dirname(__file__))))
# train_source = os.path.join(root_dir2, 'data\\cnews', 'cnews.train.txt')
# contents, labels = read_file(train_source)
# df_train = pd.DataFrame(contents, labels)
# print(df_train)
