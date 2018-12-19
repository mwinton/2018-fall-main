"""Implementation of methods for part of speech tagging using HMMs."""

import numpy as np

import collections
from collections import Counter, defaultdict

import logging
import json

def normalize_as_logp(counts):
    log_total = np.log(sum(counts.values()))
    return {k:(np.log(counts[k]) - log_total) for k in counts}


#  from scipy.misc import logsumexp
def logsumexp(a):
    """Simple re-implementation of scipy.misc.logsumexp."""
    a_max = np.max(a)
    if a_max == -np.inf:
        return -np.inf
    sumexp = np.sum(np.exp(a - a_max))
    return np.log(sumexp) + a_max


class HMM(object):

    # Fields that will be ignored when serializing / deserializing the model.
    SERIALIZE_OMITTED_FIELDS = set(['initial_counts', 'transition_counts',
                                    'emission_counts'])

    def __init__(self):
        # c(y_0): counts for y_0
        # string -> int
        self.initial_counts = Counter()

        # c(yy'): counts for (y_{i-1}, y_i)
        # string (y_{i-1}) -> string (y_i) -> int
        self.transition_counts = defaultdict(lambda: Counter())

        # c(x|y): counts for tag, word
        # string (tag) -> string (word) -> int
        self.emission_counts = defaultdict(lambda: Counter())

        # Log probabilities, computed by compute_logprobs()
        # Types are same as above, but with float values.
        # DO NOT ACCESS THESE DIRECTLY; use the helper functions below instead.
        self._initial = None     # will be string -> float
        self._transition = None  # will be string -> string -> float
        self._emission = None    # will be string -> string -> float

        # Misc bookkeeping
        self.tagset = set()
        self.id_to_tag = dict()
        self.tag_to_id = dict()

    def initial(self, tag):
        return self._initial.get(tag, -np.inf)

    def emission(self, tag, word):
        return self._emission.get(tag, {}).get(word, -np.inf)

    def transition(self, tag_1, tag_2):
        return self._transition.get(tag_1, {}).get(tag_2, -np.inf)

    def update_counts(self, tagged_sentence):
        """Accumulate counts of initial states, transitions, and emissions.

        Updates self.initial_counts, self.transition_counts, and self.emission_counts, as defined
        in the constructor above.

        These types are defined as defaultdicts and counters, so you don't need
        to initialize anything. Just do:
            self.initial_counts[tag] += 1
        or
            self.transition_counts[tag][tag] += 1
        as appropriate.

        Args:
            tagged_sentence: list((string, string)) list of (word, tag) tuples
        """

        for i, (w, t) in enumerate(tagged_sentence):
            if i == 0:
                # Sequence start count
                self.initial_counts[t] += 1
            else:
                # Transition from last state
                t_1 = tagged_sentence[i-1][1]
                self.transition_counts[t_1][t] += 1
            # Emission counts
            self.emission_counts[t][w] += 1

    def compute_logprobs(self):
        """Compute log-probabilities.

        Compute log-probabilities from the counts. Remember that self.transition
        and self.emission should be nested dicts, with the first key being the
        tag that the inner dict is conditioned on.

        Hint: use the normalize_as_logp() function, and keep this simple!
        You may want to refer back to Assignment 2, Part 1.
        """
        # Initial.
        self._initial = normalize_as_logp(self.initial_counts)

        # Transition.
        self._transition = {k: normalize_as_logp(self.transition_counts[k])
                            for k in self.transition_counts}

        # Emission.
        self._emission = {k: normalize_as_logp(self.emission_counts[k])
                          for k in self.emission_counts}

        # Compute set of all known POS tags.
        self.tagset = set(self._emission.keys())

    ##
    # Forward-backward inference
    def forward(self, sentence):
        """Run the Forward algorithm to compute alpha.

        We'll implement alpha as a dict, where the keys are tuples (i,tag) and
        the values are log-probabilities.
        So alpha[(3,'N')]  in the code is equal to log(alpha(3,'N')) in the
        equations in the writeup.

        Your alpha table should have entries for each i = 0, 1, ..., N and each
        possible tag in self.tagset.

        Args:
            sentence: list(string) sequence to tag

        Returns:
            alpha: dict((int, string) -> float) forward beliefs, as
                log-probabilities.
        """
        alpha = dict()
        #### YOUR CODE HERE ####
        # Iterate through the sentence from left to right.
        for i, w in enumerate(sentence):
            pass



        # Hint:  if you fail the unit tests, print out your alpha here
        #        and check it manually against the tests.
        #  print("cell: ", i, t, "   ")
        #  print("emission: ", self.emission(t,w), "   ")
        #  print("sum terms: ", sum_terms, "   ")
        #  print("alpha: ", alpha[(i,t)])

        #### END(YOUR CODE) ####
        return alpha

    def backward(self, sentence):
        """Run the Backward algorithm to compute beta.

        We'll implement beta as a dict, where the keys are tuples (i,tag) and
        the values are log-probabilities.
        So beta[(3,'N')]  in the code is equal to log(beta(3,'N')) in the
        equations in the writeup.

        Args:
            sentence: list(string) sequence to tag

        Returns:
            beta: dict((int, string) -> float) backward beliefs, as
                log-probabilities.
        """
        # YOUR CODE HERE
        beta = dict()


        # END(YOUR CODE)
        return beta

    def forward_backward(self, sentence, return_tables=False):
        """Determine POS tags according to forward-backward.

        Hint: use your functions for forward and backward.

        Args:
            sentence: list(string) sequence to tag
            return_tables: if true, also returns alpha and beta tables

        Returns:
            list(string), the most likely POS tag for each word, as determined
            by forward-backward.
            (optional) alpha and beta tables
        """
        tags = []
        alpha = self.forward(sentence)
        beta = self.backward(sentence)

        # For each position...
        for i in range(len(sentence)):
            # ... compute the score for each possible tag...
            candidates = [(alpha[(i,t)] + beta[(i,t)], t) for t in self.tagset]
            # ... pick the one that scores highest.
            tags.append(max(candidates)[1])

        return (tags, alpha, beta) if return_tables else tags


    def build_viterbi_delta(self, sentence):
        """Determine POS tags using the Viterbi algorithm.

        Hint:
          This code is nearly identical to the forward algorithm, except "max"
          takes the place of a summation.
          Note that in addition, you will need to keep backpointers (similar
          to best_cuts_with_trace in assignment 4) so that the sequence can be
          recovered.

        Args:
          sentence: list(string) sequence to tag

        Returns:
          delta: the "delta" table from Viterbi.
          bp: backpointers to determine which sequence generated that score.
        """

        # Viterbi table. Similar to alpha from the Forward algorithm, this is a
        # map of (i, tag) -> log probability
        delta = dict()
        # Backpointers, map of (i, tag) -> previous tag
        bp = dict()

        #### YOUR CODE HERE ####






        #### END(YOUR CODE) ####

        return delta, bp

    def viterbi(self, sentence, return_tables=False):
        """Determine POS tags using the Viterbi algorithm.

        Args:
            sentence: list(string) sequence to tag
            return_tables: if true, also returns delta and backpointer tables

        Returns:
            list(string), the most likely sequence of POS tags for the
            sentence.
            (optional) delta and backpointer tables
        """
        n = len(sentence)
        if n <= 0:
            return []

        # Build DP table.
        delta, bp = self.build_viterbi_delta(sentence)

        # Find the sequence that scores best.
        end_score, end_tag = max([(delta[(n-1,t)], t) for t in self.tagset])

        # Follow backpointers to obtain sequence with the highest score.
        tags = [end_tag]
        for i in reversed(range(0, n-1)):
            tags.append(bp[(i + 1,tags[-1])])
        tags = list(reversed(tags))

        return (tags, delta, bp) if return_tables else tags

    ####
    # Helper methods for serialization. Safe to ignore for Assignment 5.
    @staticmethod
    def from_json(s):
        """Load a model from a JSON string.

        Usage:
            with open("saved_model.json") as fd:
                model = HMM.from_json(fd.read())

        Args:
            s : (string) serialized model as JSON string.
        """
        d = json.loads(s)
        assert(d['name'] == "HMM")
        model = HMM()
        for field in model.__dict__:
            if field in model.SERIALIZE_OMITTED_FIELDS:
                continue
            if isinstance(getattr(model, field), set):
                logging.info("Loading set field '%s'" % field)
                setattr(model, field, set(d[field]))
            else:
                logging.info("Loading field '%s'" % field)
                setattr(model, field, d[field])
        return model

    def to_json(self):
        """Save model to a JSON string.

        Returns:
            (string) serialized model as JSON string.
        """
        d = dict()
        d['name'] = "HMM"
        model = self
        for field in model.__dict__:
            if field in model.SERIALIZE_OMITTED_FIELDS:
                continue
            if isinstance(getattr(model, field), set):
                logging.info("Exporting set field '%s'" % field)
                d[field] = sorted(list(getattr(model, field)))
            else:
                logging.info("Exporting field '%s'" % field)
                d[field] = getattr(model, field)
        return json.dumps(d, indent=2)
