from __future__ import print_function
from __future__ import division

import nltk
from nltk.tree import Tree
import types

def verify_ptb_install():
    # Download PTB metadata
    assert(nltk.download('ptb'))

    import hashlib
    from nltk.corpus import ptb
    # Be sure we have the category list
    assert('news' in ptb.categories())

    m = hashlib.md5()  # NOT SECURE!
    m.update(','.join(ptb.fileids()).encode('utf8'))
    if m.hexdigest() == 'e3b49c6df5529560b2945e6a4715f9b0':
        print('Penn Treebank succesfully installed!')
        return True
    else:
        print('Error installing Penn Treebank (hash mismatch).')
        print('It may still work - try loading it in NLTK.')
        return False

##
# Tree preprocessing functions
def get_np_real_child(node, **kw):
    """Find the real child, skipping over injected 'NP-*' cross-reference
    nodes.

    Args:
        node: (nltk.tree.Tree) nonterminal node

    Returns:
        list(nltk.tree.Tree) list of processed child nodes
    """
    #  if type(node) == types.UnicodeType:
    if not isinstance(node, Tree):
        return [node]
    if 'NONE' in node.label():
        return []

    real_children = []
    if node.label().startswith('NP-'):
        for child in node:
            real_children.extend(get_np_real_child(child, **kw))
    else:
        real_children.append(node)
    return [clean_tree(c, **kw) for c in real_children]
    #  return map(clean_tree, real_children)

def simplify_label(label):
    nonterminal = isinstance(label, nltk.grammar.Nonterminal)
    symbol = (label.symbol() if nonterminal else label)
    symbol = symbol.split("-",1)[0]
    symbol = symbol.split("=",1)[0]
    return nltk.grammar.Nonterminal(symbol) if nonterminal else symbol

def clean_tree(tree, **kw):
    """Make a copy of a Tree, stripping NP-* cross-reference nodes.

    If 'simplify=True' passed as a keyword argument, will also convert all
    annotated tags (e.g. S-TPC, N=2) to their basic forms.

    Args:
        tree: (nltk.tree.Tree) nonterminal node

    Returns:
        nltk.tree.Tree, a new tree with cross-reference nodes removed.
    """
    children = []
    for child in tree:
        children.extend(get_np_real_child(child, **kw))
    if kw.get('simplify', False):
        return Tree(simplify_label(tree.label()), children)
    else:
        return Tree(tree.label(), children)

