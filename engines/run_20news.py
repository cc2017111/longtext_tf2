# -*- coding: utf-8 -*-
# @Time : 2021/7/22 9:56 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : run_20news.py 
# @Software: PyCharm
from argparse import ArgumentParser
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
from transformers import BertTokenizer
from engines.configure import Configure
from engines.utils.logger import get_logger
from engines.utils.main_loop import main_loop
from engines.models import Introspector, ClassificationReasoner
from engines.data import DataManager
from engines.utils.memreplay import mem_replay
DEFAULT_MODEL_NAME = 'bert-base-chinese'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
CAPACITY = 512 # Working Memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LongText')
    parser.add_argument('--config_file', default='system.config')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)
    logger = get_logger(configs.log_dir)
    mode = configs.mode.lower()

    tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    if mode == 'train':
        logger.info('mode:train')
        main_loop(configs, logger)

    if mode == 'interactive_predict':
        print("interactive_predicting...")
        dataManger = DataManager(configs, logger)
        num_classes = dataManger.max_label_number
        introspector = Introspector(bert_path=DEFAULT_MODEL_NAME, configs=configs)
        reasoner = ClassificationReasoner(bert_path=DEFAULT_MODEL_NAME, num_classes=num_classes)

        checkpoint_introspector = tf.train.Checkpoint(model=introspector)
        checkpoint_reansoner = tf.train.Checkpoint(model=reasoner)

        introspector_checkpoint_dictionary = os.path.join(configs.checkpoints_dir, "introspector")
        reasoner_checkpoint_dictionary = os.path.join(configs.checkpoints_dir, "reasoner")
        intro_checkpoint_manager = tf.train.CheckpointManager(checkpoint_introspector,
                                                              directory=introspector_checkpoint_dictionary,
                                                              max_to_keep=5)
        reansoner_checkpoint_manager = tf.train.CheckpointManager(checkpoint_reansoner,
                                                                  directory=reasoner_checkpoint_dictionary,
                                                                  max_to_keep=5)

        intro_checkpoint_path = intro_checkpoint_manager.restore_or_initialize()
        if intro_checkpoint_path is not None:
            logger.info("restore introspector checkpoint at {}".format(intro_checkpoint_path))

        reasoner_checkpoint_path = reansoner_checkpoint_manager.restore_or_initialize()
        if reasoner_checkpoint_path is not None:
            logger.info("restore reansoner checkpoint at {}".format(reasoner_checkpoint_path))

        dataset = dataManger.get_valid_set()
        for qbuf, dbuf in tqdm(dataset):
            buf, relevance_score = mem_replay(introspector, qbuf, dbuf, times=configs.times)
            print(buf, relevance_score)
            inputs_buf = np.zeros(shape=(3, CAPACITY), dtype=np.long)
            inputs = [np.expand_dims(t, axis=0) for t in buf.export(out=(inputs_buf[0], inputs_buf[1], inputs_buf[2]))]
            output = reasoner.call(input_ids=inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2], labels=[0])
            print(output)
            break
