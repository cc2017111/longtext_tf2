# -*- coding: utf-8 -*-
# @Time : 2021/7/14 16:09 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : buffer.py 
# @Software: PyCharm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import random
from transformers import BertTokenizer


DEFAULT_MODEL_NAME = 'bert-base-chinese'
CAPACITY = 512
BLOCK_SIZE = 64


class Block:
    tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    def __init__(self, ids, pos, blk_type=1, **kwargs):
        self.ids = ids
        self.pos = pos
        self.blk_type = blk_type # 0 sentence A, 1 sentence B
        self.relevance = 0
        self.estimation = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):
        return self.blk_type < other.blk_type or (self.blk_type == other.blk_type and self.pos < other.pos)

    def __ne__(self, other):
        return self.pos != other.pos or self.blk_type != other.blk_type

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))


class Buffer:
    @staticmethod
    def split_document_into_blocks(d, tokenizer, cnt=0, properties=None):
        '''
                    d: [['word', '##piece'], ...] # a document of tokenized sentences
                    properties: [
                                    [
                                        (name: str, value: any), # len(2) tuple, sentence level property
                                        (name: str, position: int, value: any) # len(3) tuple, token level property
                                    ],
                                    []... # len(d) lists
                                ]
        '''
        ret = Buffer()
        # 遇到标点断句时的代价损失，换行无代价，逗号断句比句号、问好、感叹号大，这符合实际。
        end_tokens = {'\n': 0, '。': 1, '？': 1, '！': 1, '，': 2}
        # basic cost 遇到标点断句代价4，小于长度超过block_size的硬断句8，对应伪代码中的c。
        sen_cost, break_cost = 4, 8
        # 为长文本中所有可能断句（遇到标点，或者硬分断）的位置，计算在此断句是增加的代价损失。poses是（断句位置，增加代价）的list。
        poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]

        # 加入一个头位置
        poses.insert(0, (-1, 0))
        # 如文末没有标点，则再文末增加一个断句位置，cost为0，不增加代价。
        if poses[-1][0] < len(d) - 1:
            poses.append((len(d) - 1, 0))

        # 为poses增加硬分断位置，设置代价为8
        x = 0
        while x < len(poses) - 1:
            if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
                poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
            x += 1

        # best是最佳断句位置，计算原则是断句长度小于block size 并且在断句时代价最小，并且分段的长度尽可能长。
        # 对应伪代码，自行实现。
        best = [(0, 0)]
        for i, (p, cost) in enumerate(poses):
            if i == 0:
                continue
            best.append((-1, 100000))
            for j in range(i - 1, -1, -1):
                if p - poses[j][0] > BLOCK_SIZE:
                    break
                value = best[j][1] + cost + sen_cost
                if value < best[i][1]:
                    best[i] = (j, value)

            assert best[i][0] >= 0
        intervals, x = [], len(poses) - 1

        # intervals是根据best计算出来的断句位置（start_index，end_index）的list
        while x > 0:
            l = poses[best[x][0]][0]
            intervals.append((l + 1, poses[x][0] + 1))
            x = best[x][0]
        if properties is None:
            properties = []

        for st, en in reversed(intervals):
            cnt += 1
            tmp = d[st: en] + [tokenizer.sep_token]
            tmp_kwargs = {}
            for p in properties:
                if len(p) == 2:
                    tmp_kwargs[p[0]] = p[1]
                elif len(p) == 3:
                    if st <= p[1] < en:
                        tmp_kwargs[p[0]] = (p[1] - st, p[2])
                else:
                    raise ValueError('Invalid property {}'.format(p))
            ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs))

        return ret, cnt

    def __init__(self):
        self.blocks = []

    def __add__(self, buf):
        ret = Buffer()
        ret.blocks = self.blocks + buf.blocks
        return ret

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, key):
        return self.blocks[key]

    def __str__(self):
        return ''.join([str(b) + '\n' for b in self.blocks])

    def clone(self):
        ret = Buffer()
        ret.blocks = self.blocks.copy()
        return ret

    def calc_size(self):
        return sum([len(b) for b in self.blocks])

    def block_ends(self):
        t, ret = 0, []
        for b in self.blocks:
            t += len(b)
            ret.append(t)
        return ret

    def insert(self, b, reverse=True):
        if not reverse:
            for index in range(len(self.blocks) + 1):
                if index >= len(self.blocks) or b < self.blocks[index]:
                    self.blocks.insert(index, b)
                    break
        else:
            for index in range(len(self.blocks), -1, -1):
                if index == 0 or self.blocks[index - 1] < b:
                    self.blocks.insert(index, b)
                    break

    def merge(self, buf):
        ret = Buffer()
        t1, t2 = 0, 0
        while t1 < len(self.blocks) or t2 < len(buf):
            if t1 < len(self.blocks) and (t2 >= len(buf) or self.blocks[t1] < buf.blocks[t2]):
                ret.blocks.append(self.blocks[t1])
                t1 += 1
            else:
                ret.blocks.append(buf.blocks[t2])
                t2 += 1
        return ret

    def filtered(self, fltr: 'function blk, index->bool', need_residue=False):
        ret, ret2 = Buffer(), Buffer()
        for i, blk in enumerate(self.blocks):
            if fltr(blk, i):
                ret.blocks.append(blk)
            else:
                ret2.blocks.append(blk)
        if need_residue:
            return ret, ret2
        else:
            return ret

    def random_sample(self, size):
        assert size <= len(self.blocks)
        index = sorted(random.sample(range(len(self.blocks)), size))
        ret = Buffer()
        ret.blocks = [self.blocks[i] for i in index]
        return ret

    def sort_(self):
        self.blocks.sort()
        return self

    def fill(self, buf):
        ret, tmp_buf, tmp_size = [], self.clone(), self.calc_size()
        for blk in buf:
            if tmp_size + len(blk) > CAPACITY:
                ret.append(tmp_buf)
                tmp_buf, tmp_size = self.clone(), self.calc_size()
            tmp_buf.blocks.append(blk)
            tmp_size += len(blk)
        ret.append(tmp_buf)
        return ret

    def export(self, device=None, length=None, out=None):
        if out is None:
            if length is None:
                total_length = self.calc_size()
                if total_length > CAPACITY:
                    raise ValueError('export inputs larger than capacity')
            else:
                total_length = length * len(self.blocks)
            ids, att_masks, type_ids = np.zeros(shape=(3, total_length), dtype=np.int32)
        else: # must be zeros and big enough
            ids, att_masks, type_ids = out
            att_masks[:] = 0
        t = 0
        for b in self.blocks:
            ids[t:t + len(b)] = b.ids # id
            # if b.blk_type == 1:
            #     type_ids[t:w] = 1 # sentence B
            att_masks[t:t + len(b)] = 1 # attention_mask
            t += len(b) if length is None else length
        return ids, att_masks, type_ids

    def export_as_batch(self, device, length=BLOCK_SIZE+1, add_cls=False):
        ids, att_masks, type_ids = self.export(device, length)
        return ids.view(-1, length), att_masks.view(-1, length), type_ids.view(-1, length)

    def export_relevance(self, length=None, dtype=tf.float32, out=None):
        if out is None:
            total_length = self.calc_size() if length is None else length * len(self.blocks)
            relevance = tf.zeros(total_length, dtype=dtype)
        else:
            relevance = out
        t = 0
        for b in self.blocks:
            w = t + (len(b) if length is None else length)
            if b.relevance >= 1:
                relevance[t: w] = 1
            t = w
        return relevance


def buffer_collate(batch): # does not collate
    return batch


if __name__ == "__main__":
    s = """33岁的杭州网红女子小冉今年5月到杭州华颜医疗美容医院做吸脂填充手术，两个多月后因全身感染造成多器官衰竭，于7月13日抢救无效死亡。消息一出引发全网关注。
今天中午，杭州市卫健委发布了关于该事故的初步调查情况通报，认定这是一起医疗事故，华颜医疗美容医院存在术前缺乏认识、术中操作不当，术后观察处理不及时等过错，与患者死亡存在因果关系，承担全部责任，并已做出赔偿。
同时，西湖区卫健局对涉事的华颜医疗美容医院做出警告和罚款的处罚，责成其停业整改，对负有责任的医务人员将做出进一步处理。杭州市将对此次医疗事故举一反三，加强医疗机构执业管理，加大对医美乱象的整治。
那么，造成杭州女子死亡的吸脂手术到底是怎样一类的手术？对此，上海第九人民医院整复外科主任医师王丹茹表示，吸脂简单来讲就是采用小切口进入皮下脂肪层，通过负压将多余的脂肪抽吸出来，以达到身体塑形的效果，从技术角度本身并不难。但从小冉的创面和手术时间来看，这次手术规模不算小。而且相比手术本身来说，术前的检查、方案制定和术后的应对措施更考验医院和医生。
据企业查APP显示，杭州华颜医疗美容医院成立于2018年6月，经营范围为医疗美容科、美容外科、美容皮肤科等。近两年，这家医疗美容医院曾因病历资料不全且未见医师签名，发布违法广告等行为，受辖区卫健局、市场监管局和公安局的4次行政处罚。
近年来，随着爱美从众心理，越来越多的女性开始"迷恋"上整形美容，医疗美容服务行业蓬勃发展。据中国整形美容行业协会发布的年度报告预测，到2022年，中国整形市场规模 将达到3000亿元。
一边是市场如火如荼，一边却是行业乱象频发，王丹茹表示，眼下绝大部分的整形美容机构存有医疗技术力量跟不上、收费混乱、没有建立医疗保险共保体等问题，这也是整形美容有关医疗纠纷频频见诸报端的原因，因此，相关部门应进一步加强对医院和医生资质的审核，尤其是一些小型、私立医院对手术分级不够严谨，亟待加强监管。
事实上近年来，国家对于医美整形行业的监管力度也一直在加大。对于医疗美容行业中存在的黑医美、黑医生、黑产品等问题，今年6月10日，国家卫健委也印发了《打击非法医疗美容服务专项整治工作方案》的通知，对非法医疗美容服务开展专项整治工作。
此外，还有专家提醒，消费者也要选择资质正规的整形机构，最大限度规避风险，切莫贪图价格便宜，因小失大
"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    Buffer.split_document_into_blocks(tokenizer.tokenize(s), tokenizer)