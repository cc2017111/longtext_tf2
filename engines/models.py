import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from transformers import BertConfig, TFBertModel
from transformers import TFBertForSequenceClassification


class Introspector(Model):

    def __init__(self, bert_path, configs):
        super(Introspector, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = TFBertModel.from_pretrained(bert_path)
        self.bert.trainable = True
        self.dropout = tf.keras.layers.Dropout(configs.dropout)
        self.classifier = tf.keras.layers.Dense(1)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_ids'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='attention_mask'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='token_type_ids')])
    def call(self, input_ids, attention_mask, token_type_ids):

        embedding_outputs = self.bert(inputs=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = embedding_outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


class ClassificationReasoner(Model):

    def __init__(self, bert_path, num_classes):
        super(ClassificationReasoner, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.num_classes = num_classes
        self.bert = TFBertForSequenceClassification.from_pretrained(bert_path, num_labels=self.num_classes)
        self.bert.trainable = True
        self.softmax = tf.keras.layers.Softmax()

    @classmethod
    def export_labels(self, bufs):
        labels = np.zeros(len(bufs), dtype=np.float)
        for i, buf in enumerate(bufs):
            labels[i] = int(buf[0].label)
        return labels, [[b for b in buf if b.blk_type == 0] for buf in bufs]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_ids'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='attention_mask'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='token_type_ids'),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32, name='labels')])
    def call(self, input_ids, attention_mask, token_type_ids, labels=None):
        logits = self.bert(inputs=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_outputs = logits[0]
        softmax_outputs = self.softmax(sequence_outputs)
        result = softmax_outputs
        if labels is not None:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sequence_outputs)
            result = (loss, softmax_outputs)
        return result
