# %%
import re
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import pickle


root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)
from engines.utils.buffer import Buffer, Block
from sklearn.datasets import fetch_20newsgroups

data_train = fetch_20newsgroups(subset='train', random_state=21)
data_test = fetch_20newsgroups(subset='test', random_state=21)
DEFAULT_MODEL_NAME = 'roberta-base'

def clean(data):
    tmp_doc = []
    for words in data.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc
# %%
def process(dataset, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    cnt, batches = 0, []
    # for i in tqdm(range(len(dataset.data))):
    for i in range(20):
        d, l = clean(dataset.data[i]), dataset.target[i]
        label_name = dataset.target_names[l]
        qbuf, cnt = Buffer.split_document_into_blocks([tokenizer.cls_token], tokenizer, cnt=cnt, properties=[('label_name', label_name), ('label', l), ('_id', str(i)), ('blk_type', 0)])
        dbuf, cnt = Buffer.split_document_into_blocks(tokenizer.tokenize(d), tokenizer, cnt)
        print(dbuf)
        batches.append((qbuf, dbuf))
    with open(os.path.join(root_dir, 'data', f'20news_{dataset_name}.pkl'), 'wb') as fout:
        pickle.dump(batches, fout)
    return batches
# %%
process(data_train, 'train')
process(data_test, 'test')