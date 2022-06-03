import warnings
warnings.filterwarnings('ignore')
import argparse
import gensim.downloader as api
import numpy as np
import re
from tqdm import tqdm


from engines.data import DataManager
from engines.configure import Configure
from engines.utils.logger import get_logger


def remove_special_split(blk):
    return re.sub(r'</s>|<pad>|<s>|\w', ' ', str(blk)).lower().split()

def init_relevance(a, conditional_transformer=[]):
    print("Initialize relevance...")
    total = 0
    word_vectors = api.load("glove-wiki-gigaword-100")
    for qbuf, dbuf in tqdm(a):
        total += _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transformer)
    print(f'Initialized {total} question-document pairs!')


def _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transformer=[], threshold=0.15):
    for tranform_func in conditional_transformer:
        qbuf, dbuf = tranform_func(qbuf, dbuf)
    dvecs = []
    for blk in dbuf:
        doc = [word_vectors[w] for w in remove_special_split(blk) if w in word_vectors]
        if len(doc) > 0:
            dvecs.append(np.stack(doc))
        else:
            dvecs.append(np.zeros((1, 100)))

    qvec = np.stack([word_vectors[w] for w in remove_special_split(qbuf) if w in word_vectors])
    scores = [np.matmul(qvec, dvec.T).mean() for dvec in dvecs]
    max_score_abs = max(scores) - min(scores) + 1e-6
    for i, blk in enumerate(dbuf):
        if 1 - scores[i] / max_score_abs < threshold:
            blk.relevance = max(blk.relevance, 1)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LongText')
    parser.add_argument('--config_file', default='system.config')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)
    logger = get_logger(configs.log_dir)
    dataManger = DataManager(configs, logger)
    dataset = dataManger.get_training_set()[:50]
    init_relevance(dataset)
