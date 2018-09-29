from __future__ import print_function
from __future__ import division

import copy
import numpy as np
import tensorflow as tf

from w266_common import utils, vocabulary
import models

import unittest

class TestLayerBuilders(unittest.TestCase):
    def test_embedding_layer(self):
        with tf.Graph().as_default():
            tf.set_random_seed(10)
            ids_ = tf.constant([[0, 127, 512],
                                [63, 191,  0]], dtype=tf.int32)
            xs_ = models.embedding_layer(ids_, V=512,
                                         embed_dim=100,
                                         init_scale=1.0)
            self.assertEqual(xs_.get_shape().as_list(), [2, 3, 100])
            var_names = [v.name for v in tf.trainable_variables()]
            self.assertEqual(var_names, ["W_embed:0"])
            self.assertEqual(tf.trainable_variables("W_embed")[0].get_shape().as_list(),
                             [512, 100])

    def test_softmax_output_layer(self):
        with tf.Graph().as_default():
            tf.set_random_seed(10)
            h_ = tf.ones(shape=[3, 24], dtype=tf.float32)
            labels_ = tf.range(3)
            loss_, logits_ = models.softmax_output_layer(h_, labels_, 10)
            self.assertEqual(loss_.get_shape().as_list(), [])
            self.assertEqual(logits_.get_shape().as_list(), [3, 10])
            var_names = [v.name for v in tf.trainable_variables()]
            self.assertEqual(var_names, ["Logits/W_out:0", "Logits/b_out:0"])
            self.assertEqual(tf.trainable_variables("Logits/W_out")[0].get_shape().as_list(),
                             [24, 10])
            self.assertEqual(tf.trainable_variables("Logits/b_out")[0].get_shape().as_list(),
                             [10,])


class TestFCWithDropout(unittest.TestCase):
    def test_fc_with_dropout(self):
        with tf.Graph().as_default(), tf.Session() as sess:
            tf.set_random_seed(10)
            x_ = tf.ones(shape=[3, 24], dtype=tf.float32)
            with tf.variable_scope("Hidden"):
                h_ = models.fully_connected_layers(x_, [100], dropout_rate=0.5,
                                                   is_training=False)
            h_nz_ = tf.reduce_sum(tf.cast(tf.equal(h_, 0),
                                          dtype=tf.int32))
            with tf.variable_scope("Hidden", reuse=True):
                h_train_ = models.fully_connected_layers(x_, [100], dropout_rate=0.5,
                                                         is_training=True)
            h_train_nz_ = tf.reduce_sum(tf.cast(tf.equal(h_train_, 0),
                                                dtype=tf.int32))
            self.assertEqual(h_.get_shape().as_list(), [3, 100])
            self.assertEqual(h_train_.get_shape().as_list(), [3, 100])

            sess.run(tf.global_variables_initializer())
            h_nz, h_train_nz = sess.run([h_nz_, h_train_nz_])
            self.assertEqual(h_nz, 0)
            # Check that with dropout, number of zeros is close to expected
            expected_num_zero = 0.5 * np.prod(h_.get_shape().as_list())
            self.assertLessEqual(h_train_nz, expected_num_zero * 1.1)
            self.assertGreaterEqual(h_train_nz, expected_num_zero * 0.9)


class TestNeuralBOW(unittest.TestCase):
    def setUp(self):
        self.params = dict(V=512, embed_dim=24, hidden_dims=[12, 7],
                            num_classes=5, encoder_type='bow',
                            lr=0.1, optimizer='adagrad', beta=0.01)
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

    def test_BOW_encoder(self):
        with self._graph.as_default():
            tf.set_random_seed(10)
            ids_ = tf.constant([[0, 127, 512],
                                [63, 191,  0]], dtype=tf.int32)
            ns_ = tf.constant([3, 2], dtype=tf.int32)
            h_, xs_ = models.BOW_encoder(ids_, ns_, is_training=False, **self.params)
            self.assertEqual(h_.get_shape().as_list(), [2, 7])
            self.assertEqual(xs_.get_shape().as_list(), [2, 3, 24])

            name_to_shape = {"Embedding_Layer/W_embed:0": [512, 24],
                             "Hidden_0/kernel:0": [24, 12],
                             "Hidden_0/bias:0": [12],
                             "Hidden_1/kernel:0": [12, 7],
                             "Hidden_1/bias:0": [7]}
            names_found = set()
            for var_ in tf.trainable_variables():
                self.assertIn(var_.name, name_to_shape)
                self.assertEqual(var_.get_shape().as_list(),
                                 name_to_shape[var_.name])
                names_found.add(var_.name)
            for name in name_to_shape:
                # Check that all expected vars were covered.
                self.assertIn(name, names_found)

