import numpy as np
import os
CAPACITY = 512 # Working Memory


def score_blocks(qbuf, relevance_token):
    ends = qbuf.block_ends()
    relevance_blk = np.ones(len(ends))
    for i in range(len(ends)):
        if qbuf[i].blk_type > 0:
            relevance_blk[i] = (relevance_token[ends[i-1]:ends[i]]).numpy().mean()
    return relevance_blk


def intervention(configs, bufs, labels, crucials, loss_reansoner, reasoner):
    print("intervention...")
    max_bs = configs.batch_size
    max_blk_num = max([len(buf) for buf in bufs])
    inputs = np.zeros(shape=(4, len(bufs), CAPACITY), dtype=np.int32)
    for i in range(len(bufs)):
        bufs[i].export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        ids = inputs[0, i]
        attn_masks = inputs[1, i]
        type_ids = inputs[2, i]
        bs = len(bufs[i]) - len(crucials[i])
        ids = np.reshape(ids, newshape=(1, -1))
        ids = np.tile(ids, (bs, 1))
        type_ids = np.reshape(type_ids, newshape=(1, -1))
        type_ids = np.tile(type_ids, (bs, 1))
        attn_masks = np.reshape(attn_masks, newshape=(1, -1))
        attn_masks = np.tile(attn_masks, (bs, 1))
        label = np.reshape(labels[i], newshape=(1, -1))
        label = np.tile(label, (bs, 1))
        label = label.squeeze()
        blk_start, t = 0, 0
        for blk in bufs[i]:
            blk_end = blk_start + len(blk)
            if blk not in crucials[i]:
                attn_masks[t, blk_start:blk_end] = 0
                t += 1
            blk_start = blk_end
        assert t == bs
        losses = []
        for j in range((bs - 1) // max_bs + 1):
            l, r = max_bs * j, min(bs, max_bs * (j + 1))
            print(l, r)
            print("ids:",ids[l:r])
            # print("\nattn_masks:",attn_masks[l:r])
            # print("\ntype_ids:",type_ids[l:r])
            print("\nlabel:",label)
            result = reasoner.call(ids[l:r], attn_masks[l:r], type_ids[l:r], labels=label[l:r])
            result = result[0] if isinstance(result, tuple) else result
            print("result:", result.numpy())
            losses.append(result)
        losses_delta = np.concatenate(losses, axis=0) - loss_reansoner[i]
        print("loss_delta:", losses_delta.numpy())

        t = 0
        for blk in bufs[i]:
            print("loss_delta[t]:", losses_delta[t])
            if blk in crucials[i]:
                pass
            else:
                if losses_delta[t] >= configs.levelup_threshold and blk.relevance < 2:
                    write_changes(configs, blk, 'relevance', blk.relevance + 1)
                elif losses_delta[t] <= configs.leveldown_threshold and blk.relevance > -1:
                    write_changes(configs, blk, 'relevance', blk.relevance - 1)
                t += 1
        assert t == bs


def write_changes(configs, blk, key, value):
    with open(os.path.join(configs.tmp_dir, 'changes.txt'), 'a') as f:
        f.write('{} {} {}\n'.format(blk.pos, key, value))

