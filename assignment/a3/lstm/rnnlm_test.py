from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import rnnlm

from w266_common import vocabulary, utils

import copy
import numpy as np
import tensorflow as tf

import unittest


class TestRNNLMCore(unittest.TestCase):
    def setUp(self):
        model_params = dict(V=512, H=100, num_layers=1)
        self.lm = rnnlm.RNNLM(**model_params)
        self.lm.BuildCoreGraph()

    def test_shapes_embed(self):
        self.assertEqual(self.lm.W_in_.get_shape().as_list(), [512, 100])

    def test_shapes_recurrent(self):
        self.assertEqual(self.lm.cell_.state_size[0].c, 100)
        self.assertEqual(self.lm.cell_.state_size[0].h, 100)
        init_c_shape = self.lm.initial_h_[0].c.get_shape().as_list()
        init_h_shape = self.lm.initial_h_[0].h.get_shape().as_list()
        self.assertEqual(init_c_shape, [None, 100])
        self.assertEqual(init_h_shape, [None, 100])

        self.assertEqual(self.lm.final_h_[0].c.get_shape().as_list(),
                         init_c_shape)
        self.assertEqual(self.lm.final_h_[0].h.get_shape().as_list(),
                         init_h_shape)

    def test_shapes_output(self):
        self.assertEqual(self.lm.W_out_.get_shape().as_list(), [100, 512])
        self.assertEqual(self.lm.b_out_.get_shape().as_list(), [512])
        self.assertEqual(self.lm.loss_.get_shape().as_list(), [])


class TestRNNLMTrain(unittest.TestCase):
    def setUp(self):
        model_params = dict(V=512, H=100, num_layers=1)
        self.lm = rnnlm.RNNLM(**model_params)
        self.lm.BuildCoreGraph()
        self.lm.BuildTrainGraph()

    def test_shapes_train(self):
        self.assertEqual(self.lm.train_loss_.get_shape().as_list(), [])
        self.assertNotEqual(self.lm.loss_, self.lm.train_loss_)
        self.assertIsNotNone(self.lm.train_step_)


class TestRNNLMSampler(unittest.TestCase):
    def setUp(self):
        model_params = dict(V=512, H=100, num_layers=1)
        self.lm = rnnlm.RNNLM(**model_params)
        self.lm.BuildCoreGraph()
        self.lm.BuildSamplerGraph()

    def test_shapes_sample(self):
        self.assertEqual(self.lm.pred_samples_.get_shape().as_list(),
                         [None, None, 1])

class RunEpochTester(unittest.TestCase):
    def __init__(self, *args, run_epoch_fn=None, score_dataset_fn=None, **kw):
        super(RunEpochTester, self).__init__(*args, **kw)
        self.run_epoch = run_epoch_fn
        self.score_dataset = score_dataset_fn

    def setUp(self):
        sequence = ["a", "b", "c", "d"]
        self.vocab = vocabulary.Vocabulary(sequence)
        ids = self.vocab.words_to_ids(sequence)
        self.train_ids = np.array(ids * 50000, dtype=int)
        self.test_ids = np.array(ids * 100, dtype=int)

        model_params = dict(V=self.vocab.size, H=10,
                            softmax_ns=2, num_layers=1)
        self.lm = rnnlm.RNNLM(**model_params)
        self.lm.BuildCoreGraph()
        self.lm.BuildTrainGraph()
        self.lm.BuildSamplerGraph()
        # For toy model, ignore sampled softmax.
        self.lm.train_loss_ = self.lm.loss_

    def injectCode(self, run_epoch_fn, score_dataset_fn):
        self.run_epoch = run_epoch_fn
        self.score_dataset = score_dataset_fn

    def test_toy_model(self):
        if self.run_epoch is None or self.score_dataset is None:
            self.skipTest("RunEpochTester: run_epoch_fn and score_dataset_fn "
                          "must be provided.")

        self.assertIsNotNone(self.run_epoch)
        self.assertIsNotNone(self.score_dataset)

        with tf.Session(graph=self.lm.graph) as sess:
            tf.set_random_seed(42)
            sess.run(tf.global_variables_initializer())
            bi = utils.rnnlm_batch_generator(self.train_ids, 5, 10)
            self.run_epoch(self.lm, sess, bi, learning_rate=0.01,
                           train=True, verbose=True, tick_s=1.0)
            train_loss = self.score_dataset(self.lm, sess, self.train_ids,
                                            name="Train set")
            test_loss = self.score_dataset(self.lm, sess, self.test_ids,
                                           name="Test set")
        # This is a *really* simple dataset, so you should have no trouble
        # getting almost perfect scores.
        self.assertFalse(train_loss is None)
        self.assertFalse(test_loss is None)
        self.assertLessEqual(train_loss, 0.1)
        self.assertLessEqual(test_loss, 0.2)


if __name__ == '__main__':
  unittest.main()
