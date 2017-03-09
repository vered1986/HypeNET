import codecs
import bsddb

from docopt import docopt


def main():
    """
    Creates a knowledge resource from triplets file: first step,
    receives the entire triplets file and saves the following files:
    '_path_to_id.db', '_id_to_path.db', '_term_to_id.db', '_id_to_term.db'
    """

    # Get the arguments
    args = docopt("""Creates a knowledge resource from triplets file: first step,
    receives the entire triplets file and saves the following files:
    '_path_to_id.db', '_id_to_path.db', '_term_to_id.db', '_id_to_term.db'

    Usage:
        create_resource_from_corpus_1.py <frequent_paths_file> <terms_file> <resource_prefix>

        <frequent_paths_file> = the file containing the frequent paths, that should be included in the resource.
        Similarly to Snow et al. (2004), we considered only paths that occurred with 5 different term-pairs in the
        corpus.
        <terms_file> = the file containing all the terms.
        <resource_prefix> = the file names' prefix for the resource files
    """)

    frequent_paths_file = args['<frequent_paths_file>']
    terms_file = args['<terms_file>']
    resource_prefix = args['<resource_prefix>']

    # Load the frequent paths
    print 'Saving the paths...'
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

    frequent_paths = None

    # Load the terms
    print 'Saving the terms...'
    with codecs.open(terms_file, 'r', 'utf-8') as f_in:
        terms = [line.strip() for line in f_in]

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


if __name__ == '__main__':
    main()
