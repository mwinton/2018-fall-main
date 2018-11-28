import pcfg

import collections
from nltk import Tree
import numpy as np
import unittest

class TestPCFG(unittest.TestCase):
    
    def test_pcfg(self):
        o = pcfg.PCFG()
        tree = Tree('S', (Tree('NP', ('foo',)), Tree('VP', ('bar',))))

        o.update_counts(tree)
        self.assertSetEqual(
                set([(p, 1) for p in tree.productions()]),
                set(o.production_counts.items()))
        self.assertSetEqual(set([(p.lhs(), 1) for p in tree.productions()]),
                set(o.lhs_counts.items()))
        o.update_counts(tree)

        tree = Tree('S', (Tree('VP', ('foo',)), Tree('NP', ('bar',))))
        o.update_counts(tree)
        o.update_counts(tree)
        self.assertEqual(6, len(o.production_counts))
        for count in o.production_counts.values():
            self.assertEqual(2, count)
        self.assertEqual(3, len(o.lhs_counts))
        for count in o.lhs_counts.values():
            self.assertEqual(4, count)

        o.compute_scores()
        for production, score in o.scored_productions.items():
            self.assertAlmostEqual(-0.69314718055, score, msg='%s' % production)
