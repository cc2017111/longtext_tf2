# -*- coding: utf-8 -*-
# @Time : 2021/7/22 10:39 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : main_loop.py 
# @Software: PyCharm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import math
import tensorflow as tf
from tqdm import tqdm
from engines.data import DataManager
from engines.utils.shuffle_util import data_shuffle
from engines.utils.score_blocks import score_blocks, intervention
from engines.utils.block_interface import BlockPositionInterface
from engines.initialize_relevance import init_relevance
from engines.models import Introspector, ClassificationReasoner
DEFAULT_MODEL_NAME = 'bert-base-chinese'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main_loop(configs, logger):
    os.makedirs(configs.tmp_dir, exist_ok=True)
    dataManger = DataManager(configs, logger)

    dataset = dataManger.get_training_set()
    interface = BlockPositionInterface(dataset)
    num_classes = dataManger.max_label_number
    if configs.init_relevance is not None:
        if hasattr(configs, 'conditional_transformer'):
            ct = configs.conditional_transformer
            del configs.conditional_transformer
        else:
            ct = []
        init_relevance(dataset, conditional_transformer=ct)

    if configs.optimizer == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(configs.learning_rate)
    elif configs.optimizer == 'Adadelta':
        optimizer = tf.keras.optimizers.Adadelta(configs.learning_rate)
    elif configs.optimizer == 'RMSProp':
        optimizer = tf.keras.optimizers.RMSprop(configs.learning_rate)
    elif configs.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(configs.learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(configs.learning_rate)

    print("start initialize model...")
    introspector = Introspector(bert_path=DEFAULT_MODEL_NAME, configs=configs)
    reasoner = ClassificationReasoner(bert_path=DEFAULT_MODEL_NAME, num_classes=num_classes)

    introspector_checkpoint_dictionary = os.path.join(configs.checkpoints_dir, "introspector")
    reasoner_checkpoint_dictionary = os.path.join(configs.checkpoints_dir, "reasoner")
    checkpoint_introspector = tf.train.Checkpoint(optimizer=optimizer, model=introspector)
    checkpoint_reansoner = tf.train.Checkpoint(optimizer=optimizer, model=reasoner)

    intro_checkpoint_manager = tf.train.CheckpointManager(checkpoint_introspector, directory=introspector_checkpoint_dictionary,
                                                          max_to_keep=5)
    reansoner_checkpoint_manager = tf.train.CheckpointManager(checkpoint_reansoner, directory=reasoner_checkpoint_dictionary,
                                                              max_to_keep=5)
    for i in range(configs.epoch):
        start_time = time.time()
        intro_dataset = interface.build_random_buffer(num_samples='1,1,1,1')
        num_iterations = int(math.ceil(1.0 * len(intro_dataset) / configs.batch_size))
        intro_dataset_shuffled = data_shuffle(intro_dataset)
        logger.info('epoch: {}/{}'.format(i + 1, configs.epoch))
        train_introspector_loss = 0
        train_reasoner_loss = 0
        intro_checkpoint_path = intro_checkpoint_manager.restore_or_initialize()
        if intro_checkpoint_path is not None:
            logger.info("restore introspector checkpoint at {}".format(intro_checkpoint_path))
        for iteration in tqdm(range(num_iterations)):
        # for iteration in range(10):
            buf_batch_intro, x_train_batch_intro, y_train_batch_intro, att_mask_train_batch_intro, token_type_ids_train_batch_intro = dataManger.next_batch(intro_dataset_shuffled, start_index=iteration * configs.batch_size)
            with tf.GradientTape() as tape_introspector:
                intro_outputs = introspector.call(input_ids=x_train_batch_intro, attention_mask=att_mask_train_batch_intro,
                                                  token_type_ids=token_type_ids_train_batch_intro)
                if y_train_batch_intro is not None:
                    if att_mask_train_batch_intro is not None:
                        active_loss = tf.reshape(att_mask_train_batch_intro, shape=[-1]) == 1
                        active_logits = tf.reshape(intro_outputs, shape=[-1])[active_loss]
                        active_labels = tf.cast(tf.reshape(y_train_batch_intro, shape=[-1])[active_loss], dtype=tf.float32)
                        intro_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=active_labels, logits=active_logits)
                    else:
                        intro_losses = tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(intro_outputs, shape=[-1]), tf.reshape(y_train_batch_intro, shape=[-1]))
                intro_loss = tf.reduce_mean(intro_losses)

            # print("intro_loss:", intro_loss)
            gradients = tape_introspector.gradient(intro_loss, introspector.trainable_variables)
            optimizer.apply_gradients(zip(gradients, introspector.trainable_variables))
            train_introspector_loss += intro_loss

            if iteration % configs.print_per_batch == 0:
                logger.info("training introspector batch: %5d, loss: %.5f" % (iteration, train_introspector_loss/configs.print_per_batch))
                train_introspector_loss = 0

            for j, buf in enumerate(buf_batch_intro):
                for j, blk in enumerate(buf):
                    with open(os.path.join(configs.tmp_dir, 'estimations.txt'), 'a') as f:
                        f.write(f'{blk.pos} {score_blocks(buf, intro_losses)[j].item()}\n')
        intro_checkpoint_manager.save()
        tf.saved_model.save(introspector, os.path.join(configs.model_save_dir, 'introspector'))


        logger.info("training reasoner at epoch {}".format(i))
        interface.collect_estimations_from_dir(configs.tmp_dir)
        reason_dataset = interface.build_promising_buffer(num_samples='1,1,1,1')
        num_iterations = int(math.ceil(1.0 * len(reason_dataset) / configs.batch_size))
        reason_dataset_shuffled = data_shuffle(reason_dataset)
        reasoner_checkpoint_path = reansoner_checkpoint_manager.restore_or_initialize()
        if reasoner_checkpoint_path is not None:
            logger.info("restore reansoner checkpoint at {}".format(reasoner_checkpoint_path))

        for iteration in tqdm(range(num_iterations)):
        # for iteration in range(3):
            buf_batch_reason, x_train_batch_reason, y_train_batch_reason, att_mask_train_batch_reason, token_type_ids_train_batch_reason = dataManger.next_batch(
                reason_dataset_shuffled, start_index=iteration * configs.batch_size)
            labels, crucials = reasoner.export_labels(buf_batch_reason)
            with tf.GradientTape() as tape_reasoner:
                reasoner_outputs = reasoner.call(input_ids=x_train_batch_reason, attention_mask=att_mask_train_batch_reason,
                                                 token_type_ids=token_type_ids_train_batch_reason, labels=labels)
                reasoner_losses = reasoner_outputs[0] if isinstance(reasoner_outputs, tuple) else reasoner_outputs
                reasoner_loss = tf.reduce_mean(reasoner_losses)
            print("reasoner_losses:", reasoner_losses.numpy())
            gradients = tape_reasoner.gradient(reasoner_loss, reasoner.trainable_variables)
            optimizer.apply_gradients(zip(gradients, reasoner.trainable_variables))
            train_reasoner_loss += reasoner_loss

            # print("reasoner_loss:", reasoner_loss)
            if iteration % configs.print_per_batch == 0 and iteration != 0:
                # print("loss:", train_reasoner_loss)
                logger.info("training reasoner batch: %5d, loss: %.5f" % (iteration, train_reasoner_loss/configs.print_per_batch))
                train_reasoner_loss = 0

            if configs.latent:
                intervention(configs, buf_batch_reason, labels, crucials, reasoner_losses, reasoner)

        reansoner_checkpoint_manager.save()
        tf.saved_model.save(reasoner, os.path.join(configs.model_save_dir, 'reasoner'))
        if configs.latent and i > 1:
            interface.apply_changes_from_dir(configs.tmp_dir)

