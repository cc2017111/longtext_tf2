import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tqdm import tqdm
import logging
import random
import numpy as np
from engines.utils.buffer import Buffer
CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = 'bert-base-chinese'
BLOCK_SIZE = 63 # The max length of an episode


class BlockPositionInterface:
    def __init__(self, dataset):
        assert isinstance(dataset, list)
        self.d = {}
        self.dataset = dataset
        for bufs in dataset:
            for buf in bufs:
                for blk in buf:
                    assert blk.pos not in self.d
                    self.d[blk.pos] = blk

    def set_property(self, pos, key, value=None):
        blk = self.d[pos]
        if value is not None:
            setattr(blk, key, value)
        elif hasattr(blk, key):
            delattr(blk, key)

    def apply_changes_from_dir(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('changes'):
                self.apply_changes_from_file(filename)
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def apply_changes_from_file(self, filename):
        with open(filename, 'r') as fin:
            for line in fin:
                tmp = [int(s) if s.isdigit() or s[0] == '-' and s[1:].isdigit() else s for s in line.split()]
                self.set_property(*tmp)

    def collect_estimations_from_dir(self, tmp_dir):
        ret = []
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('estimations'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        self.d[pos].estimation = estimation
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def build_random_buffer(self, num_samples):
        n0, n1 = [int(s) for s in num_samples.split(',')[:2]]
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        logging.info('building buffers for introspection...')
        for qbuf, dbuf in tqdm(self.dataset):
            lb = max_blk_num - len(qbuf)
            st = random.randint(0, max(0, len(qbuf) - lb * n0))
            for i in range(n0):
                buf = Buffer()
                buf.blocks = qbuf.blocks + dbuf.blocks[st + i * lb:st + (i+1) * lb]
                ret.append(buf)

            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            for i in range(n1):
                selected_pblks = random.sample(pbuf.blocks, min(lb, len(pbuf)))
                selected_nblks = random.sample(nbuf.blocks, min(lb - len(selected_pblks), len(nbuf)))
                buf = Buffer()
                buf.blocks = qbuf.blocks + selected_pblks + selected_nblks
                ret.append(buf.sort_())
        return ret

    def build_promising_buffer(self, num_samples):
        n2, n3 = [int(x) for x in num_samples.split(',')[2:]]
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        logging.info("building buffers for reasoning...")
        for qbuf, dbuf in tqdm(self.dataset):
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            if len(pbuf) >= max_blk_num - len(qbuf):
                pbuf = pbuf.random_sample(max_blk_num - len(qbuf) - 1)
            lb = max_blk_num - len(qbuf) - len(pbuf)
            estimations = [blk.estimation for blk in nbuf]
            keeped_indices = np.argsort(estimations)[::-1][:n2 * lb]
            # sorted(estimations, reverse=True)
            # estimations = estimations[:n2 * lb]
            selected_nblks = [blk for i, blk in enumerate(nbuf) if i in keeped_indices]
            while 0 < len(selected_nblks) < n2 * lb:
                selected_nblks = selected_nblks * (n2 * lb // len(selected_nblks) + 1)
            for i in range(n2):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + selected_nblks[i * lb: (i+1) * lb]
                ret.append(buf.sort_())
            for i in range(n3):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + random.sample(nbuf.blocks, min(len(nbuf), lb))
                ret.append(buf.sort_())
        return ret

