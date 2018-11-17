import pos

import collections
import numpy as np
#  from scipy.misc import logsumexp
from pos import logsumexp  # use same implementation as tagger

import unittest

class TestTagging(unittest.TestCase):

    def process_sentences(self, sentences):
        processed_sentences = []
        for sentence in sentences:
            processed = [tuple(s.split(':',1)) for s in sentence.split()]
            processed_sentences.append(processed)
        return processed_sentences
    
    def test_process_sentences(self):
        sentences = ['A:B C:D', 'E:F']
        expected = [[('A', 'B'), ('C', 'D')], [('E', 'F')]]
        self.assertEqual(expected, self.process_sentences(sentences))

    def test_estimate_probabilities(self):
        testdata = [
            'James:N ate:V the:DT food:N',
            'John:N danced:V a:DT lively:JJ jig:N',
            'the:DT contagious:JJ flu:N',
            'a:DT language:N'
          ]
        tagged_sentences = self.process_sentences(testdata)

        hmm = pos.HMM()
        for s in tagged_sentences:
            hmm.update_counts(s)
        hmm.compute_logprobs()

        # Initial checks.
        self.assertEqual(2, len(hmm._initial))
        self.assertAlmostEqual(-np.log(2), hmm.initial('DT'))
        self.assertAlmostEqual(-np.log(2), hmm.initial('N'))
        self.assertNotIn('V', hmm._initial)

        # Transition checks.
        self.assertEqual(4, len(hmm._transition))
        self.assertEqual({
            'N': {'V': 0.0},
            'V': {'DT': 0.0},
            'JJ': {'N': 0.0},
            'DT': {'N': -np.log(2), 'JJ':-np.log(2)}}, hmm._transition)

        # Emission checks. (required biannually by state of California)
        self.assertEqual(4, len(hmm._emission))
        self.assertAlmostEqual(-np.log(2), hmm.emission('DT','a'))
        self.assertAlmostEqual(-np.log(2), hmm.emission('DT','the'))

    def test_compute_logprobs_initial_counts(self):
        hmm = pos.HMM()
        hmm.initial_counts = {'V': 10, 'N': 22}
        hmm.transition_counts = {'DT': {'V': 2, 'N': 10}, 'N': {'DT': 1, 'N': 7}}
        hmm.emission_counts = {'DT': {'the': 84}}
        hmm.compute_logprobs()

        # Initial normalization.
        self.assertEqual(2, len(hmm._initial))
        self.assertIn('N', hmm._initial)
        self.assertIn('V', hmm._initial)
        self.assertAlmostEqual(np.log(10. / (10. + 22.)), hmm.initial('V'))
        self.assertAlmostEqual(np.log(22. / (10. + 22.)), hmm.initial('N'))

        # Transition normalization.
        self.assertEqual(2, len(hmm._transition))
        self.assertSetEqual(set(hmm._transition.keys()), set(['DT', 'N']))
        self.assertAlmostEqual(np.log(2. / (2. + 10.)), hmm.transition('DT','V'))
        self.assertAlmostEqual(np.log(10. / (2. + 10.)), hmm.transition('DT','N'))

        # Emission normalization.
        self.assertEqual(1, len(hmm._emission))
        self.assertIn('DT', hmm._emission)
        self.assertAlmostEqual(hmm.emission('DT','the'), 0.0)  # np.log(1.0) == 0.0

    def test_alpha(self):
        hmm = pos.HMM()
        hmm._initial = {'V': -2, 'DT': -4}
        hmm._transition = {'DT': {'V': -5, 'N': -7}, 'N': {'DT': -11, 'N': -13}, 'V': {'N': -8}}
        hmm._emission = {'DT': {'the': -17}, 'N': {'potato': -19, 'the': -23}, 'V': {'potato': -14, 'the': -21}}
        hmm.tagset = set(['DT', 'N', 'V'])

        alpha = hmm.forward(['the', 'potato'])

        # Only these four keys are possible.  Your code should either:
        # a) simply not create other keys (and treat missing ones as impossible)
        # b) create them, but ensure they end up as -inf (Remember: log P(x) = -inf, if P(x) = 0.)
        expected = {
            # Hint:
            # P(initial == DT) x P(DT emits 'the'), taking the log for numerical stability
            # is log P(DT) + log P(DT -> 'the') = -4 - 17.
            (0, 'DT'): -4 -17,
            (0, 'V'): -2 -21,
            (1, 'N'): -19 + logsumexp([-21 -7, -23 -8]),
            (1, 'V'): -21-5-14}

        for key, value in sorted(expected.items()):
            self.assertAlmostEqual(value, alpha[key])

        extra_keys = set(alpha.keys()) - set(expected.keys())
        for key in extra_keys:
            self.assertEqual(alpha[key], float('-inf'), msg='Key:{}'.format(key))


    def test_beta(self):
        hmm = pos.HMM()
        hmm._initial = {'V': -2, 'DT': -4}
        hmm._transition = {'DT': {'V': -5, 'N': -7}, 'N': {'DT': -11, 'N': -13}}
        hmm._emission = {'DT': {'the': -17}, 'N': {'potato': -19, 'the': -23}, 'V': {'potato': -14, 'the': -50}}
        hmm.tagset = set(['DT', 'N', 'V'])

        beta = hmm.backward(['the', 'potato'])

        # Similar to forward above, either avoid creating impossible keys or make sure they have a value of -inf.
        expected = {
            (1, 'DT'): 0.0,
            (1, 'V'): 0.0,
            (1, 'N'): 0.0,
            (0, 'N'): -13 -19,
            (0, 'DT'): logsumexp([-5 -14, -7 -19])}

        for key, value in expected.items():
            self.assertAlmostEqual(value, beta[key], msg='Key:{}'.format(key))

        extra_keys = set(beta.keys()) - set(expected.keys())
        for key in extra_keys:
            self.assertEqual(beta[key], float('-inf'), msg='Key:{}'.format(key))


    def test_forward_backward(self):
        hmm = pos.HMM()
        hmm._initial = {'V': -20, 'DT': -4}
        hmm._transition = {'DT': {'V': -5, 'N': -7}, 'N': {'DT': -11, 'N': -13}, 'V': {'N': -8}}
        hmm._emission = {'DT': {'the': -17}, 'N': {'potato': -2, 'the': -23}, 'V': {'potato': -14, 'the': -50}}
        hmm.tagset = set(['DT', 'N', 'V'])

        self.assertListEqual(hmm.forward_backward(['the', 'potato']), ['DT', 'N'])


    def test_build_viterbi_delta(self):
        hmm = pos.HMM()
        hmm._initial = {'V': -2, 'DT': -4}
        hmm._transition = {'DT': {'V': -5, 'N': -7}, 'N': {'DT': -11, 'N': -13}, 'V': {'N': -8}}
        hmm._emission = {'DT': {'the': -17}, 'N': {'potato': -19, 'the': -23}, 'V': {'potato': -14, 'the': -50}}
        hmm.tagset = set(['DT', 'N', 'V'])

        delta, bp = hmm.build_viterbi_delta(['the', 'potato'])

        # Validate scores for index 0.
        expected = {
            (0, 'DT'): (-4 -17, None),
            (0, 'V'): (-52, None),
            (1, 'N'): (-19 + max([-21 -7, -52 -8]), 'DT'),
            (1, 'V'): (-21-5-14, 'DT')}

        for key, value in expected.items():
          self.assertIn(key, delta)
          self.assertEqual(delta[key], value[0], msg='Delta for {}'.format(key))
          if key[0] == 0:
              self.assertTrue(key not in bp or bp[key] == None)
          else:
              self.assertIn(key, bp)
              self.assertEqual(bp[key], value[1], msg='BP for {}'.format(key))

        extra_keys = set(delta.keys()) - set(expected.keys())
        for key in extra_keys:
            self.assertEqual(delta[key], float('-inf'), msg='Key:{}'.format(key))


    def test_viterbi_end_to_end(self):
        hmm = pos.HMM()
        hmm._initial = {'V': -20, 'DT': -4}
        hmm._transition = {'DT': {'V': -5, 'N': -7}, 'N': {'DT': -11, 'N': -13}, 'V': {'N': -8}}
        hmm._emission = {'DT': {'the': -17}, 'N': {'potato': -2, 'the': -23}, 'V': {'potato': -14, 'the': -50}}
        hmm.tagset = set(['DT', 'N', 'V'])

        self.assertListEqual(hmm.viterbi(['the', 'potato']), ['DT', 'N'])


if __name__ == '__main__':
    unittest.main()
