# An elegant weapon, for a more civilized age
from __future__ import division

import numpy as np
from collections import Counter, defaultdict
import json

class PCFG(object):
    """Simple implementation of a Probabilistic Context-Free Grammar.

    This uses NLTK data structures, and is similar to nltk.grammar.PCFG but
    with more efficient lookups and the added advantage of you, the student,
    getting to write key parts of the implementation!
    """

    def __init__(self):
        # Map of nltk.grammar.Production -> int
        self.production_counts = Counter()
        # Map of nltk.grammar.Nonterminal -> int
        self.lhs_counts = Counter()

        # See compute_scores
        self.scored_productions = None
        # See build_index
        self.parsing_index = None

    def top_productions(self, n=10):
        return self.production_counts.most_common(n)

    def top_lhs(self, n=10):
        return self.lhs_counts.most_common(n)

    def update_counts(self, parsed_sentence):
        """Accumulate counts of productions from a single sentence.

        Updates self.production_counts and self.lhs_counts, incrementing counts
        by 1 for each production seen and each lhs seen.

        Args:
            parsed_sentence: nltk.tree.Tree

        Returns:
            None
        """
        pass
        #### YOUR CODE HERE ####



        #### END(YOUR CODE) ####

    def compute_scores(self):
        """Compute log-probabilities.

        Populate self.scored_productions, which has the same keys as
        self.production_counts but where the values are the log probabilities
        log(p) = log(numerator) - log(denominator), according to the equation
        in the notebook.
        """
        # Map of nltk.grammar.Production -> float
        self.scored_productions = dict()

        #### YOUR CODE HERE ####



        #### END(YOUR CODE) ####

    def build_index(self):
        """Index productions by RHS, for use in bottom-up parsing.

        This should be run after compute_scores()
        """
        # Map of tuple(nltk.grammar.Nonterminal) ->
        #                  list((nltk.grammar.Nonterminal, double))
        # Maps from RHS to (LHS, score)
        self.parsing_index = defaultdict(list)
        for production in self.scored_productions:
            score = self.scored_productions[production]
            l = self.parsing_index[production.rhs()]
            l.append((production.lhs(), score))
        # Remove defaultdict behavior
        self.parsing_index.default_factory = None


    def lookup_rhs(self, *rhs):
        """Lookup candidates by RHS.

        Usage: grammar.lookup(left, right)
        (use this instead of directly accessing
            grammar.parsing_index[(left, right)] )

        Args:
            *rhs: rhs elements (word type or nonterminal)

        Returns:
            list((nonterminal, double)) of compatible LHS and their scores
        """
        return self.parsing_index.get(rhs, [])

    @staticmethod
    def from_json(s):
        """Load a model from a JSON string.

        Usage:
            with open("saved_model.json") as fd:
                model = PCFG.from_json(fd.read())

        Args:
            s : (string) serialized model as JSON string.
        """
        d = json.loads(s)
        assert(d['name'] == "PCFG")
        model = PCFG()
        # De-serialize from list of 2-element lists
        # Don't bother re-converting to nltk.grammar.Nonterminal
        model.parsing_index = {tuple(k):v for (k,v) in d['parsing_index']}
        return model

    def to_json(self):
        """Save model to a JSON string.

        Only save parsing_index, since this is the only structure needed for
        CKY parsing.

        Returns:
            (string) serialized model as JSON string.
        """
        from nltk.grammar import Nonterminal
        d = dict()
        d['name'] = "PCFG"
        model = self
        # Serialize index as list of tuples, since JSON can't have non-string
        # keys.
        d['parsing_index'] = model.parsing_index.items()

        class NonterminalEncoder(json.JSONEncoder):
            """Custom JSON encoder to serialize nltk.grammar.Nonterminal."""
            def default(self, obj):
                if isinstance(obj, Nonterminal):
                    return obj.symbol()
                return json.JSONEncoder.default(self, obj)

        return json.dumps(d, indent=2, cls=NonterminalEncoder)

