##
# Helpers to export a chart from cky.py

def _process_ij(ij):
    return "%d:%d" % ij

def _process_symbol(symbol):
    return str(symbol)

def _process_tree(ij, head, tree):
    ret = {'score': tree.logprob(),
           'head': _process_symbol(head),
           'i': ij[0], 'j': ij[1]}
    # Handle preterminal cells
    if len(tree) == 1:
        ret['preterminal'] = True
    elif len(tree) == 2:
        ret['preterminal'] = False
        ret['left'] = _process_symbol(tree[0].label())
        ret['right'] = _process_symbol(tree[1].label())
        # Count leaves to get split offset
        ret['split'] = ij[0] + len(tree[0].leaves())
    else:
        raise ValueError("tree must have 1 or 2 children. Found %d in tree %s"
                         % (len(tree), tree.pprint()))
    return ret


def chart_to_simple_dict(chart):
    """Process the chart data structure into a serializable form.

    Outer keys (i,j) become "i:j"
    Inner keys s become str(s)
    Tree values become dicts with fields:
        score: float
        preterminal: bool
        left, right: string (nonterminals only)
        split: int (nonterminals only)

    Args:
        chart: map from (int, int) -> symbol -> ProbabilisticTree

    Returns:
        simple chart: map from string -> string -> dict
    """
    ret = {_process_ij(ij): {
                _process_symbol(symbol): _process_tree(ij, symbol, tree)
                for symbol, tree in chart[ij].iteritems()
            }
            for ij in chart}
    # Add sort key list, for simple rendering of top candidates
    # while preserving fast lookup for backtrace.
    for ij in ret:
        d = ret[ij]
        sd = sorted(d.items(), key=lambda (k,v): -1*v['score'])
        d['__sorted_symbols__'] = [k for (k,v) in sd]
    return ret
