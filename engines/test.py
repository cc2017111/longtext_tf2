# -*- coding: utf-8 -*-
# @Time : 2021/7/26 10:35 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : test.py 
# @Software: PyCharm
import pickle
import logging
import os
import numpy as np
import tensorflow as tf
import argparse
from engines.data import DataManager
from engines.configure import Configure
from engines.utils.logger import get_logger
from transformers import BertTokenizer
CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = 'bert-base-chinese'


parser = argparse.ArgumentParser(description='LongText')
parser.add_argument('--config_file', default='system.config')
args = parser.parse_args()
configs = Configure(config_file=args.config_file)
logger = get_logger(configs.log_dir)
dataManger = DataManager(configs, logger)
dataset = dataManger.get_valid_set()

# times='3,5'
# times = [int(x) for x in times.split(',')]
# for k, inc in enumerate(times):
#     print(k, inc)


# labels = tf.convert_to_tensor(np.array([3]))
# print(labels.shape)
# # [2 3 1 0]
# logits = tf.constant([[-1.1258, -1.1524, -0.2506, -0.4339,  0.5988]])
# print(logits.shape)
#
# tfloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# print(tfloss.numpy())

# a = np.array([1, 2, 3])
# a = np.unsqueeze(a)
#
# print(a)

#
data = "我在CBA打球".lower()
# print(data)
# tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
# data1 = tokenizer.tokenize(data)
# print(data1)