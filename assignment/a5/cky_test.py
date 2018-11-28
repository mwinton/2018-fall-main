import cky
import pcfg

import collections
import numpy as np
import unittest

class MockGrammar(pcfg.PCFG):
    def __init__(self, parsing_index):
        self.parsing_index = collections.defaultdict(list)
        self.parsing_index.update(parsing_index)

class TestParsing(unittest.TestCase):

    def test_failing_rule_application(self):
        chart = cky.make_chart()
        grammar = MockGrammar({
            ('the',): [('DT', -4)],
            ('potato',): [('N', -3), ('V', -20)],})

        parser = cky.CKYParser(grammar)
        self.assertFalse(parser._apply_preterminal_rules(
                'the rock'.split(), chart))

    def finite_keys(self, cell):
        return set([pos for pos, pt in cell.items()
            if not np.isinf(pt.logprob())])

    def test_rule_application(self):
        chart = cky.make_chart()
        grammar = MockGrammar({
            ('the',): [('DT', -4)],
            ('potato',): [('N', -3), ('V', -20)],
            ('DT', 'N'): [('NP', -1), ('VP', -300)],
            ('DT', 'V'): [('VP', -2)]})

        parser = cky.CKYParser(grammar)

        sentence = 'the potato'.split()

        # Verify preterminal rule application.
        self.assertTrue(parser._apply_preterminal_rules(
            sentence, chart))

        self.assertSetEqual(set([(0, 1), (1, 2)]), set(chart.keys()))
        self.assertSetEqual(set(['DT']), self.finite_keys(chart[(0, 1)]))
        self.assertEqual(chart[(0, 1)]['DT'].label(), 'DT')
        self.assertEqual(chart[(0, 1)]['DT'].logprob(), -4)
        self.assertEqual(len(chart[(0, 1)]['DT'].leaves()), 1)
        self.assertEqual(chart[(0, 1)]['DT'].leaves()[0], 'the')

        self.assertSetEqual(set(['N', 'V']), set(chart[(1, 2)].keys()))
        self.assertEqual(chart[(1, 2)]['N'].label(), 'N')
        self.assertEqual(chart[(1, 2)]['N'].logprob(), -3)
        self.assertEqual(len(chart[(1, 2)]['N'].leaves()), 1)
        self.assertEqual(chart[(1, 2)]['N'].leaves()[0], 'potato')

        self.assertEqual(chart[(1, 2)]['V'].label(), 'V')
        self.assertEqual(chart[(1, 2)]['V'].logprob(), -20)
        self.assertEqual(len(chart[(1, 2)]['V'].leaves()), 1)
        self.assertEqual(chart[(1, 2)]['V'].leaves()[0], 'potato')

        # Verify binary rule application.
        parser._apply_binary_rules(len(sentence), chart)
        self.assertSetEqual(set([(0, 1), (1, 2), (0, 2)]), set(chart.keys()))
        self.assertEqual(self.finite_keys(chart[(0, 2)]), set(['NP', 'VP']))
        self.assertEqual(chart[(0, 2)]['NP'].label(), 'NP')
        self.assertEqual(chart[(0, 2)]['NP'].logprob(), -8)
        self.assertEqual(chart[(0, 2)]['VP'].label(), 'VP')
        self.assertEqual(chart[(0, 2)]['VP'].logprob(), -26)
        self.assertEqual(chart[(0, 2)]['VP'][1].label(), 'V')

    def test_rule_application_three_words(self):
        chart = cky.make_chart()
        grammar = MockGrammar({
            ('James',): [('N', -4)],
            ('was',): [('V', -3), ('DT', -20)],
            ('hungry',): [('JJ', -5), ('N', -21)],
            ('N', 'JJ'): [('NP', -1), ('VP', -300)],
            ('N', 'V'): [('NP', -300)],
            ('N', 'DT'): [('NP', -51)],
            ('V', 'JJ'): [('VP', -7)],
            ('N', 'VP'): [('S', -1)],
            ('JJ', 'N'): [('NP', -2)]})

        parser = cky.CKYParser(grammar)

        sentence = 'James was hungry'.split()

        # Verify preterminal rule application.
        self.assertTrue(parser._apply_preterminal_rules(
                sentence, chart))

        self.assertSetEqual(set([(0, 1), (1, 2), (2, 3)]), set(chart.keys()))
        self.assertSetEqual(set(['N']), self.finite_keys(chart[(0, 1)]))
        self.assertEqual(chart[(0, 1)]['N'].label(), 'N')
        self.assertEqual(chart[(0, 1)]['N'].logprob(), -4)
        self.assertEqual(len(chart[(0, 1)]['N'].leaves()), 1)
        self.assertEqual(chart[(0, 1)]['N'].leaves()[0], 'James')

        self.assertSetEqual(set(['DT', 'V']), self.finite_keys(chart[(1, 2)]))
        self.assertEqual(chart[(1, 2)]['V'].label(), 'V')
        self.assertEqual(chart[(1, 2)]['V'].logprob(), -3)
        self.assertEqual(len(chart[(1, 2)]['V'].leaves()), 1)
        self.assertEqual(chart[(1, 2)]['V'].leaves()[0], 'was')

        self.assertEqual(chart[(1, 2)]['DT'].label(), 'DT')
        self.assertEqual(chart[(1, 2)]['DT'].logprob(), -20)
        self.assertEqual(len(chart[(1, 2)]['DT'].leaves()), 1)
        self.assertEqual(chart[(1, 2)]['DT'].leaves()[0], 'was')

        self.assertSetEqual(set(['N', 'JJ']), self.finite_keys(chart[(2, 3)]))
        self.assertEqual(chart[(2, 3)]['JJ'].label(), 'JJ')
        self.assertEqual(chart[(2, 3)]['JJ'].logprob(), -5)
        self.assertEqual(len(chart[(2, 3)]['JJ'].leaves()), 1)
        self.assertEqual(chart[(2, 3)]['JJ'].leaves()[0], 'hungry')

        self.assertEqual(chart[(2, 3)]['N'].label(), 'N')
        self.assertEqual(chart[(2, 3)]['N'].logprob(), -21)
        self.assertEqual(len(chart[(2, 3)]['N'].leaves()), 1)
        self.assertEqual(chart[(2, 3)]['N'].leaves()[0], 'hungry')

        # Verify binary rule application.
        parser._apply_binary_rules(len(sentence), chart)
        self.assertSetEqual(
                set([(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (0, 3)]),
                set(chart.keys()))
        self.assertEqual(self.finite_keys(chart[(0, 2)]), set(['NP']))
        self.assertEqual(self.finite_keys(chart[(1, 3)]), set(['VP']))
        self.assertEqual(self.finite_keys(chart[(0, 3)]), set(['S']))
        self.assertEqual(chart[(0, 2)]['NP'].logprob(), -4 -20 -51)
        self.assertEqual(chart[(1, 3)]['VP'].logprob(), -3 -5 -7)
        self.assertEqual(chart[(0, 3)]['S'].logprob(), -4 -3 -5 -7 -1)


if __name__ == '__main__':
    unittest.main()
