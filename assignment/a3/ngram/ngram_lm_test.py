from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import ngram_lm

import copy
import numpy as np
import unittest


class TestAddKTrigramLM(unittest.TestCase):
    def setUp(self):
        self.lm = ngram_lm.AddKTrigramLM('there be here there be dragons'.split())

    def test_counts(self):
        self.assertEqual(3, len(self.lm.counts))
        self.assertSetEqual(set([
            ('there', 'be'),
            ('be', 'here'),
            ('here', 'there')]), set(self.lm.counts.keys()))
        self.assertDictEqual({
            'here': 1.0,
            'dragons': 1.0}, self.lm.counts[('there', 'be')])

    def test_words(self):
        self.assertSetEqual(set([
            'there', 'be', 'here', 'dragons']),
            set(self.lm.words))
        self.assertEqual(4, self.lm.V)

    def test_context_totals(self):
        self.assertEqual(2, self.lm.context_totals[('there', 'be')])
        self.assertEqual(1, self.lm.context_totals[('be', 'here')])

    def test_next_word_proba_no_smoothing(self):
        self.lm.set_live_params(k=0.0)

        unseen_context_error_msg = """
LM with k=0 should either crash on unseen context with a ZeroDivisionError, or
return a plausible alternative probability estimate. If the latter, please
justify your choice in your code."""
        try:
            p = self.lm.next_word_proba('w266', ['hello', 'world'])
            self.assertTrue(np.isclose(1.0/self.lm.V, p) or np.isclose(0.0, p),
                            msg=unseen_context_error_msg)
        except Exception as e:
            self.assertIsInstance(e, ZeroDivisionError,
                                  msg=unseen_context_error_msg)

        pp = self.lm.next_word_proba('w266', ['there', 'be'])
        self.assertTrue(isinstance(pp, float))

        self.assertAlmostEqual(0,
                self.lm.next_word_proba('w266', ['there', 'be']))
        self.assertAlmostEqual(0.5,
                self.lm.next_word_proba('dragons', ['there', 'be']))
        self.assertAlmostEqual(1.0,
                self.lm.next_word_proba('be', ['here', 'there']))

    def test_next_word_proba_k_exists(self):
        self.lm.set_live_params(k=10.0)

        pp = self.lm.next_word_proba('w266', ['there', 'be'])
        self.assertTrue(isinstance(pp, float))

        self.assertAlmostEqual(10. / 40.,
                self.lm.next_word_proba('w266', ['hello', 'world']))
        self.assertAlmostEqual(11. / 42.,
                self.lm.next_word_proba('dragons', ['there', 'be']))

    def test_no_mutate_on_predict(self):
        self.lm.set_live_params(k=10.0)

        lm_copy = copy.deepcopy(self.lm)

        _ = self.lm.next_word_proba('w266', ['hello', 'world'])
        _ = self.lm.next_word_proba('dragons', ['there', 'be'])

        self.assertEqual(lm_copy, self.lm,
                         msg="lm_copy != self.lm. Calls to next_word_proba " +
                         "should not modify language model parameters!")


class TestKNTrigramLM(unittest.TestCase):
    def setUp(self):
        self.lm = ngram_lm.KNTrigramLM('there be here dragons there be dragons'.split())

    def test_counts(self):
        # unigram + bigram + trigram keys.
        self.assertEqual(1 + 4 + 4, len(self.lm.counts))
        self.assertSetEqual(set([
            (),
            ('there',), ('be',), ('here',), ('dragons',),
            ('there', 'be'),
            ('be', 'here'),
            ('here', 'dragons'),
            ('dragons', 'there')]), set(self.lm.counts.keys()))
        self.assertDictEqual({
            'here': 1.0,
            'dragons': 1.0}, self.lm.counts[('there', 'be')])
        self.assertDictEqual({
            'here': 1.0,
            'there': 2.0,
            'be': 2.0,
            'dragons': 2.0}, self.lm.counts[()])

    def test_words(self):
        self.assertSetEqual(set([
            'there', 'be', 'here', 'dragons']),
            set(self.lm.words))
        self.assertEqual(4, self.lm.V)

    def test_type_contexts(self):
        self.assertSetEqual(set(['be', 'here']), self.lm.type_contexts['dragons'])
        self.assertSetEqual(set(['there']), self.lm.type_contexts['be'])
        self.assertSetEqual(set(['dragons']), self.lm.type_contexts['there'])

    def test_context_totals(self):
        self.assertEqual(2, self.lm.context_totals[('there', 'be')])
        self.assertEqual(1, self.lm.context_totals[('be', 'here')])

    def test_context_nnz(self):
        self.assertEqual(1, self.lm.context_nnz[('here', 'dragons')])
        self.assertEqual(2, self.lm.context_nnz[('there', 'be')])
        self.assertEqual(2, self.lm.context_nnz[('be',)])
        self.assertEqual(4, self.lm.context_nnz[()])

    def test_type_fertility(self):
        self.assertEqual(2, self.lm.type_fertility['dragons'])
        self.assertEqual(1, self.lm.type_fertility['there'])
        self.assertEqual(1, self.lm.type_fertility['be'])

    def test_z_tf(self):
        self.assertAlmostEqual(5.0, self.lm.z_tf)

    def test_kn_interp(self):
        self.assertAlmostEqual(0.02,
                self.lm.kn_interp('hi', ('there', 'be'), 0.1, 0.2))
        self.assertAlmostEqual(0.47,
                self.lm.kn_interp('dragons', ('there', 'be'), 0.1, 0.2))

    def test_next_word_proba(self):
        self.assertAlmostEqual(0.44375,
                self.lm.next_word_proba('dragons', ('there', 'be')))
        self.assertAlmostEqual(0.1125,
                self.lm.next_word_proba('be', ('there', 'be')))

    def test_no_mutate_on_predict(self):
        """Don't allow modifications to the LM during inference.

        This test makes sure that you aren't modifying your language model
        in the next_word_proba function. Notably, this means you shouldn't
        be adding values to any dictionaries, even by accident! In practice,
        unless you're doing online learning (not to be confused with an
        online degree program :)), you don't want to change your model except
        during the training phase.
        (Hint: use dict.get())."""
        self.lm.set_live_params(delta=0.75)

        lm_copy = copy.deepcopy(self.lm)

        _ = self.lm.next_word_proba('dragons', ('there', 'be'))
        _ = self.lm.next_word_proba('be', ('there', 'be'))

        self.assertEqual(lm_copy, self.lm,
                         msg="lm_copy != self.lm. Calls to next_word_proba " +
                         "should not modify language model parameters!")

if __name__ == '__main__':
    unittest.main()
