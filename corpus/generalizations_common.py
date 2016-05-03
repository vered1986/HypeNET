import re
import itertools
from collections import defaultdict

edge_pattern = re.compile(r'^([<>]?)([^/\s]+)/([^/]+)/([^<>\s]+)([<>]?)$', re.U)


def generalize_path(path):
    """
    Returns all the generalized paths of path: replace every word by its POS tag or a wild card.
    :param path: the original path
    :return: the generalized path set
    """

    replacements = defaultdict(set)
    edges = path.split('_')

    for edge in edges:

        replacements[edge].add(edge)

        m = edge_pattern.match(edge)
        if m and m.group(2) != 'X' and m.group(2) != 'Y':
            replacements[edge].add(m.group(3))
            replacements[edge].add('*')

    paths = set(['_'.join(path_edges) for path_edges in itertools.product(*[replacements[edge] for edge in edges])])
    return paths
