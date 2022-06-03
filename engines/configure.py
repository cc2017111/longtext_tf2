# -*- coding: utf-8 -*-
# @Time : 2021/7/22 10:14 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : configure.py 
# @Software: PyCharm
class Configure:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)
        # Training Settings
        the_item = 'pretrained_model_name'
        if the_item in config:
            self.pretrained_model_name = config[the_item]

        # Status
        the_item = 'mode'
        if the_item in config:
            self.mode = config[the_item]

        # Init relevance
        the_item = 'init_relevance'
        if the_item in config:
            self.init_relevance = self.str2none(config[the_item])

        # Without relevance labels
        the_item = 'latent'
        if the_item in config:
            self.latent = self.str2bool(config[the_item])

        # levelup_threshold
        the_item = 'levelup_threshold'
        if the_item in config:
            self.levelup_threshold = float(config[the_item])

        # leveldown_threshold
        the_item = 'leveldown_threshold'
        if the_item in config:
            self.leveldown_threshold = float(config[the_item])

        # times
        the_item = 'times'
        if the_item in config:
            self.times = config[the_item]

        # Datasets(Input/Output)
        the_item = 'datasets_fold'
        if the_item in config:
            self.datasets_fold = config[the_item]

        the_item = 'train_file'
        if the_item in config:
            self.train_file = config[the_item]

        the_item = 'dev_file'
        if the_item in config:
            self.dev_file = self.str2none(config[the_item])
        else:
            self.dev_file = None

        the_item = 'delimiter'
        if the_item in config:
            self.delimiter = config[the_item]

        the_item = 'vocabs_dir'
        if the_item in config:
            self.vocabs_dir = config[the_item]

        the_item = 'log_dir'
        if the_item in config:
            self.log_dir = config[the_item]

        the_item = 'tmp_dir'
        if the_item in config:
            self.tmp_dir = config[the_item]

        the_item = 'checkpoints_dir'
        if the_item in config:
            self.checkpoints_dir = config[the_item]

        the_item = 'model_save_dir'
        if the_item in config:
            self.model_save_dir = config[the_item]

        the_item = 'measuring_metrics'
        if the_item in config:
            self.measuring_metrics = config[the_item]

        the_item = 'embedding_dim'
        if the_item in config:
             self.embedding_dim = int(config[the_item])

        the_item = 'max_sequence_length'
        if the_item in config:
            self.max_sequence_length = int(config[the_item])
        if self.max_sequence_length > 512:
            raise Exception('the max sequence length over 512 in Bert mode')

        the_item = 'hidden_size'
        if the_item in config:
            self.hidden_size = int(config[the_item])

        the_item = 'CUDA_VISIBLE_DEVICES'
        if the_item in config:
            self.CUDA_VISIBLE_DEVICES = config[the_item]

        the_item = 'seed'
        if the_item in config:
            self.seed = int(config[the_item])

        the_item = 'is_early_stop'
        if the_item in config:
            self.is_early_stop = self.str2bool(config[the_item])

        the_item = 'patient'
        if the_item in config:
            self.patient = int(config[the_item])

        the_item = 'epoch'
        if the_item in config:
            self.epoch = int(config[the_item])

        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'dropout'
        if the_item in config:
            self.dropout = float(config[the_item])

        the_item = 'learning_rate'
        if the_item in config:
            self.learning_rate = float(config[the_item])

        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]

        the_item = 'checkpoint_name'
        if the_item in config:
            self.checkpoints_name = config[the_item]

        the_item = 'checkpoints_max_to_keep'
        if the_item in config:
            self.checkpoints_max_to_keep = int(config[the_item])

        the_item = 'print_per_batch'
        if the_item in config:
            self.print_per_batch = int(config[the_item])

    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        fins = open(input_file, mode='r', encoding='utf-8').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == '#':
                continue
            if '=' in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                try:
                    if item in config:
                        print('Warning: duplicated config item found: {}, updated.'.format((pair[0])))
                    if value[0] == '[' and value[-1] == ']':
                        value_items = list(value[1:-1].split(','))
                        config[item] = value_items
                    else:
                        config[item] = value
                except Exception:
                    print('configuration parsing error, please check correctness of the config file.')
                    exit(1)
        return config

    @staticmethod
    def str2none(string):
        if string == 'None' or string == 'none' or string == 'NONE':
            return None
        else:
            return string

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False
