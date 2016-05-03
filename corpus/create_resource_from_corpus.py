import codecs
import bsddb

from docopt import docopt
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
        <frequent_paths_file> = the file containing the frequent paths, that should be included in the resource
        <resource_prefix> = the file names' prefix for the resource files
    """)

    triplets_file = args['<triplets_file>']
    frequent_paths_file = args['<frequent_paths_file>']
    resource_prefix = args['<resource_prefix>']

    # Load the frequent paths
    with codecs.open(frequent_paths_file, 'r', 'utf-8') as f_in:
        frequent_paths = set([line.strip() for line in f_in])

    # Load the corpus
    with codecs.open(triplets_file, 'r', 'utf-8') as f_in:
        triplets = [tuple(line.strip().split('\t')) for line in f_in]

    left, right, paths = zip(*triplets)
    paths = list(sorted(set(paths).intersection(set(frequent_paths))))
    entities = list(sorted(set(left).union(set(right))))
    term_to_id = { t : i for i, t in enumerate(entities) }
    path_to_id = { p : i for i, p in enumerate(paths) }

    # Terms
    term_to_id_db = bsddb.btopen(resource_prefix + '_term_to_id.db', 'c')
    id_to_term_db = bsddb.btopen(resource_prefix + '_id_to_term.db', 'c')

    for term, id in term_to_id.iteritems():
        id, term = str(id), str(term)
        term_to_id_db[term] = id
        id_to_term_db[id] = term

    term_to_id_db.sync()
    id_to_term_db.sync()

    # Paths
    path_to_id_db = bsddb.btopen(resource_prefix + '_path_to_id.db', 'c')
    id_to_path_db = bsddb.btopen(resource_prefix + '_id_to_path.db', 'c')

    for path, id in path_to_id.iteritems():
        id, path = str(id), str(path)
        path_to_id_db[path] = id
        id_to_path_db[id] = path

    path_to_id_db.sync()
    id_to_path_db.sync()

    # Relations
    l2r_db = bsddb.btopen(resource_prefix + '_l2r.db', 'c')

    num_line = 0

    # Load the triplets file
    l2r_edges = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

    paths = set(paths)
    with codecs.open(triplets_file) as f_in:
        for line in f_in:
            try:
                x, y, path = line.strip().split('\t')
            except:
                print line
                continue

            # Frequent path
            if path in paths:
                x_id, y_id, path_id = term_to_id.get(x, -1), term_to_id.get(y, -1), path_to_id.get(path, -1)
                if x_id > -1 and y_id > -1 and path_id > -1:
                    l2r_edges[x_id][y_id][path_id] += 1

            num_line += 1
            if num_line % 1000000 == 0:
                print 'Processed ', num_line, ' lines.'

    for x in l2r_edges.keys():
        for y in l2r_edges[x].keys():
            l2r_db[str(x) + '###' + str(y)] = ','.join(
                [':'.join((str(p), str(val))) for (p, val) in l2r_edges[x][y].iteritems()])

    l2r_db.sync()


if __name__ == '__main__':
    main()
