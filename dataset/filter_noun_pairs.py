import codecs
import re
import random
import math

from docopt import docopt
from knowledge_resource import KnowledgeResource

NEG_POS_RATIO = 4


def main():
    """
    Filter out pairs from the dataset, to keep only those with enough path occurrences in the corpus
    """

    # Get the arguments
    args = docopt("""Filter out pairs from the dataset, to keep only those with enough path occurrences in the corpus

    Usage:
        filter_noun_pairs.py <corpus_prefix> <dataset_file> <min_occurrences>

        <corpus_prefix> = the corpus' resource file prefix
        <dataset_file> = the original dataset file
        <min_occurrences> = the minimum required occurrences in different paths
    """)

    corpus_prefix = args['<corpus_prefix>']
    dataset_file = args['<dataset_file>']
    min_occurrences = int(args['<min_occurrences>'])

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = KnowledgeResource(corpus_prefix)
    print 'Done!'

    # Load the dataset
    print 'Loading the dataset...'
    dataset = load_dataset(dataset_file)

    # Filter noun-pairs: keep only noun pairs that occurred with at least 5 unique paths
    print 'Filtering out noun-pairs...'
    filtered_pairs = filter_noun_pairs(corpus, dataset.keys(), min_occurrences)
    positives = [(x, y) for (x, y) in filtered_pairs if dataset[(x, y)] == 'True' ]
    negatives = [(x, y) for (x, y) in filtered_pairs if dataset[(x, y)] == 'False' ]

    if len(negatives) > len(positives) * NEG_POS_RATIO:
        filtered_pairs = random.sample(negatives, len(positives) * NEG_POS_RATIO) + positives
    else:
        filtered_pairs = random.sample(positives, int(math.ceil(len(negatives) / NEG_POS_RATIO))) + negatives

    random.shuffle(filtered_pairs)

    with codecs.open(dataset_file.replace('.tsv', '_filtered.tsv'), 'w', 'utf-8') as f_out:
        print >> f_out, '\n'.join(['\t'.join([x, y, dataset[(x, y)]]) for (x, y) in filtered_pairs])

    print 'Done!'


def load_dataset(dataset_file):
    """
    Load the dataset
    :param dataset_file:
    :return:
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        lines = [tuple(line.strip().split('\t')) for line in f_in]
        dataset = { (x, y) : label for (x, y, label) in lines }

    return dataset


def filter_noun_pairs(corpus, dataset_keys, min_occurrences):
    """
    Filter out pairs from the dataset, to keep only those with enough path occurrences in the corpus
    :param corpus: the corpus resource object
    :param dataset_keys: the (x,y) pairs in the dataset
    :param min_occurrences: the minimum required occurrences in different paths
    :return:
    """
    first_filter = [(x, y) for (x, y) in dataset_keys if len(x) > 3 and len(y) > 3 and
                    len(set(x.split(' ')).intersection(y.split(' '))) == 0]
    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in first_filter]
    no_sat_pattern = re.compile('^X/.*Y/[^/]+/[^/]+$')
    paths_x_to_y = [set(get_paths(corpus, x_id, y_id, no_sat_pattern)) for (x_id, y_id) in keys]
    filtered_keys = [first_filter[i] for i, key in enumerate(keys) if len(paths_x_to_y[i]) >= min_occurrences]

    return filtered_keys


def get_paths(corpus, x, y, pattern):
    """
    Returns the paths between (x, y) term-pair
    :param corpus: the corpus resource object
    :param x: the X entity
    :param y: the Y entity
    :param pattern: path patterns to exclude (e.g. satellites)
    :return: all paths between (x, y) which do not match the pattern
    """
    x_to_y_paths = corpus.get_relations(x, y)
    paths = [path_id for path_id in x_to_y_paths.keys() if pattern.match(corpus.get_path_by_id(path_id))]
    return paths


if __name__ == '__main__':
    main()