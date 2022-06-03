# -*- coding: utf-8 -*-
# @Time : 2021/7/25 19:02 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : data.py 
# @Software: PyCharm
class DataManager:
    def __init__(self, configs, logger):
        self.config = configs
        self.logger = logger
        self.train_file = configs.datasets_fold + '/' + configs.train_file

        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

    def get_training_set(self, train_val_ratio=0.9):
        df_train =
