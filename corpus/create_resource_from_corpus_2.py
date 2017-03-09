import codecs
import bsddb

from docopt import docopt
from itertools import count
from collections import defaultdict


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Creates a knowledge resource from triplets file.

    Usage:
        create_resource_from_corpus.py <triplets_file> <frequent_paths_file> <resource_prefix>

        <triplets_file> = the file that contains text triplets, formated as X\tY\tpath
        <frequent_paths_file> = the file containing the frequent paths, that should be included in the resource.
        Similarly to Snow et al. (2004), we considered only paths that occurred with 5 different term-pairs in the
        corpus. These could be computed using the triplet files created from parse_wikipedia.py (e.g. parsed_corpus):
        sort -u parsed_corpus | cut -f3 -d$'\t' > paths
        awk -F$'\t' '{a[$1]++; if (a[$1] == 5) print $1}' paths > frequent_paths
        <resource_prefix> = the file names' prefix for the resource files
    """)

    triplets_file = args['<triplets_file>']
    frequent_paths_file = args['<frequent_paths_file>']
    resource_prefix = args['<resource_prefix>']

    # Load the frequent paths
    with codecs.open(frequent_paths_file, 'r', 'utf-8') as f_in:
        frequent_paths = set([line.strip() for line in f_in])

    # Save the paths
    path_to_id = { path : i for i, path in enumerate(list(frequent_paths)) }
    path_to_id_db = bsddb.btopen(resource_prefix + '_path_to_id.db', 'c')
    id_to_path_db = bsddb.btopen(resource_prefix + '_id_to_path.db', 'c')

    for path, id in path_to_id.iteritems():
        id, path = str(id), str(path)
        path_to_id_db[path] = id
        id_to_path_db[id] = path

    path_to_id_db.sync()
    id_to_path_db.sync()

    # Load the terms
    terms = set()
    with codecs.open(triplets_file, 'r', 'utf-8') as f_in:
        for line in f_in:
            left, right, _ = line.strip().split('\t')
            terms.add(left)
            terms.add(right)

    # Save the terms
    term_to_id = { term : i for i, term in enumerate(terms) }
    term_to_id_db = bsddb.btopen(resource_prefix + '_term_to_id.db', 'c')
    id_to_term_db = bsddb.btopen(resource_prefix + '_id_to_term.db', 'c')

    for term, id in term_to_id.iteritems():
        id, term = str(id), str(term)
        term_to_id_db[term] = id
        id_to_term_db[id] = term

    term_to_id_db.sync()
    id_to_term_db.sync()

    # path_to_id_db = bsddb.btopen(resource_prefix + '_path_to_id.db')
    # frequent_paths = set(path_to_id_db.keys())
    #
    # l2r_edges = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    #
    # num_line = 0
    #
    # # Load the triplet file
    # with codecs.open(triplets_file) as f_in:
    #     for line in f_in:
    #         try:
    #             x, y, path = line.strip().split('\t')
    #         except:
    #             print line
    #             continue
    #
    #         # Frequent path
    #         if path in frequent_paths:
    #             x_id, y_id, path_id = term_to_id[x], term_to_id[y], path_to_id_db[path]
    #             l2r_edges[x_id][y_id][path_id] += 1
    #
    #         num_line += 1
    #         if num_line % 1000000 == 0:
    #             print 'Processed ', num_line, ' lines.'
    #
    # # Relations
    # l2r_db = bsddb.btopen(resource_prefix + '_l2r.db', 'c')
    #
    # for x in l2r_edges.keys():
    #     for y in l2r_edges[x].keys():
    #         l2r_db[str(x) + '###' + str(y)] = ','.join(
    #             [':'.join((str(p), str(val))) for (p, val) in l2r_edges[x][y].iteritems()])
    #
    # l2r_db.sync()


if __name__ == '__main__':
    main()
