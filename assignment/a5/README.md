# Assignment 5: Parsing

This assignment is optional, ungraded, and therefore has no deadline.

It has one part:

* [Parsing](CKY.ipynb)

If you get stuck, take a look at the unit tests (and, failing that, post publicly on Piazza... no need to make these private as they aren't graded!).

### Visualizations (optional)

We've built interactive [visualizations](https://cky-dot-nlp-visualizations.appspot.com/cky?sentence=James+ate+the+food) of this assignment, which we hope you'll find useful as you work through your implementations CKY. In particular, you can inspect partial hypotheses (subtrees), and see how the backpointers trace through the chart.

_Note: the grammar in the demo is simplified to reduce visual clutter, so don't be surprised if numbers don't match with your implementation!_

### The Penn Treebank (optional)

NLTK includes the `treebank` corpus, which is an abbreviated sample (3900 sentences) of the full (73k sentence) corpus.

The full corpus is available through Berkeley for research and academic purposes. We've included a copy in `ptb.zip` in this directory, along with a script that will install it to the proper directory for NLTK to access. Run as:
```
./install_ptb.sh
```
If it installs successfully, you can substitute `nltk.corpus.ptb` for `nltk.corpus.treebank`, and most functions should work normally - but with access to much more data. See [NLTK - Parsed Corpora](http://www.nltk.org/howto/corpus.html#parsed-corpora) for more information.

**NOTE:** be sure that your code passes the tests using the `treebank` corpus first.

## Submission instructions

As always, you should commit your changes often with `git add` and `git commit`.

Please submit by running the submit script:
```
./assignment/submit.sh -u your-github-username -a 5
```
You can view your work in your usual submission repo on the `a5-submit` branch.

Since this assignment isn't graded, there is no deadline.  The instructors won't know when to look at your assignment (assuming you want feedback).  If you do want us to take a look, send us a note to the **instructor email list**.
