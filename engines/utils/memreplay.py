import numpy as np
import tensorflow as tf
from engines.utils.score_blocks import score_blocks
from engines.utils.buffer import Buffer
CAPACITY = 512 # Working Memory


def mem_replay(introspector, qbuf, dbuf, times='3,5', batch_size_inference=16):
    times = [int(x) for x in times.split(',')]
    inputs = np.zeros(shape=(3, batch_size_inference, CAPACITY), dtype=np.long)
    B_set = []
    for k, inc in enumerate(times):
        num_to_keep = len(qbuf) + inc
        estimations = np.zeros(shape=len(dbuf))
        bufs, t = qbuf.fill(dbuf), 0
        print("bufs[0]:", bufs[0])
        print("bufs[1]:", bufs[1])
        for i in range((len(bufs) - 1) // batch_size_inference + 1):
            l, r = batch_size_inference * i, min(len(bufs), batch_size_inference * (i + 1))
            for j, buf in enumerate(bufs[l:r]):
                buf.export(out=(inputs[0, j], inputs[1, j], inputs[2, j]))
            logits = introspector.call(input_ids=inputs[0, :r-l], attention_mask=inputs[1, :r-l], token_type_ids=inputs[2, :r-l])
            sigmoid_logits = tf.nn.sigmoid(logits)
            for j, buf in enumerate(bufs[l:r]):
                estimation = score_blocks(buf, sigmoid_logits[j])[len(qbuf):]
                estimations[t: t + len(estimation)] = estimation
                t += len(estimation)
        assert t == len(dbuf)

        indices = np.argsort(estimations)[::-1]
        print("indices:", indices)
        qbuf_size = qbuf.calc_size()
        for idx in indices:
            if qbuf_size + len(dbuf[idx]) > CAPACITY:
                break
            if dbuf[idx] in B_set:
                continue
            qbuf_size += len(dbuf[idx])
            qbuf.insert(dbuf[idx])

        print("qbuf_1:", qbuf)
        qbuf.export(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))
        relevance_token = tf.nn.sigmoid(introspector.call(input_ids=inputs[0, :1], attention_mask=inputs[1, :1], token_type_ids=inputs[2, :1]))
        relevance_token = tf.reshape(relevance_token, shape=(-1))
        print(relevance_token.numpy())
        print(relevance_token.numpy().shape)
        relevance_blk = score_blocks(qbuf, relevance_token)
        keeped_indices = np.argsort(relevance_blk)[::-1]
        print("keeped_indices_1:", keeped_indices)
        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[:num_to_keep]
        else:
            return qbuf, relevance_blk

        print("keeped_indices_2:", keeped_indices)
        filtered_qbuf, filtered_relevance_blk = Buffer(), []
        for i, blk in enumerate(qbuf):
            if i in keeped_indices:
                filtered_qbuf.blocks.append(blk)
                # print("filtered_qbuf:", filtered_qbuf)
                filtered_relevance_blk.append(relevance_blk[i])
        qbuf = filtered_qbuf
        print("qbuf_2:", qbuf)
        B_set = [blk for blk in qbuf if blk.blk_type == 1]

    return filtered_qbuf, filtered_relevance_blk