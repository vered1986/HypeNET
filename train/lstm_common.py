import codecs

import numpy as np
from itertools import groupby

EMBEDDINGS_DIM = 50


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


def get_paths(corpus, x, y):
    """
    Get the paths that connect x and y in the corpus
    :param corpus: the corpus' resource object
    :param x:
    :param y:
    :return:
    """
    x_to_y_paths = corpus.get_relations(x, y)
    y_to_x_paths = corpus.get_relations(y, x)
    paths = { corpus.get_path_by_id(path) : count for (path, count) in x_to_y_paths.iteritems() }
    paths.update({ corpus.get_path_by_id(path).replace('X/', '@@@').replace('Y/', 'X/').replace('@@@', 'Y/') : count
                   for (path, count) in y_to_x_paths.iteritems() })
    return paths


def vectorize_path(path, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the path
    :param path:
    :param lemma_index:
    :param pos_index:
    :param dep_index:
    :param dir_index:
    :return:
    """
    path_edges = [vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index) for edge in path.split('_')]
    if None in path_edges:
        return None
    else:
        return tuple(path_edges)


def vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the edge: concatenate lemma/pos/dep and add direction symbols
    :param edge:
    :param lemma_index:
    :param pos_index:
    :param dep_index:
    :param dir_index:
    :return:
    """
    direction = ' '

    # Get the direction
    if edge.startswith('<') or edge.startswith('>'):
        direction = 's' + edge[0]
        edge = edge[1:]
    elif edge.endswith('<') or edge.endswith('>'):
        direction = 'e' + edge[-1]
        edge = edge[:-1]

    try:
        lemma, pos, dep = edge.split('/')
    except:
        return None

    return tuple([lemma_index[lemma], pos_index[pos], dep_index[dep], dir_index[direction]])


def reconstruct_edge((lemma, pos, dep, direction),
                     lemma_inverted_index, pos_inverted_index, dep_inverted_index, dir_inverted_index):
    """
    Return a string representation of the edge
    :param lemma_inverted_index:
    :param pos_inverted_index:
    :param dep_inverted_index:
    :param dir_inverted_index:
    :return:
    """
    edge = '/'.join([lemma_inverted_index[lemma], pos_inverted_index[pos], dep_inverted_index[dep]])
    dir = dir_inverted_index[direction]

    if dir[0] == 's':
        edge = dir[1] + edge
    elif dir[0] == 'e':
        edge = edge + dir[1]

    return edge


def output_predictions(predictions_file, relations, predictions, test_keys, test_labels):
    """
    Output the model predictions for the test set
    :param predictions_file: the output file path
    :param relations: the ordered list of relations
    :param predictions: the predicted labels for the test set
    :param test_keys: an ordered list of the test set (x, y) term-pairs
    :param test_labels: an ordered list of the test set labels
    :return:
    """
    with codecs.open(predictions_file, 'w', 'utf-8') as f_out:
        for i, (x, y) in enumerate(test_keys):
            print >> f_out, '\t'.join([x, y, relations[test_labels[i]], relations[predictions[i]]])


def load_embeddings(file_name, vocabulary):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :param vocabulary: limited vocabulary to load vectors for
    :return: the vocabulary and the word vectors
    """
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        words, vectors = zip(*[line.strip().split(' ', 1) for line in f_in])
    vectors = np.loadtxt(vectors)

    unknown = np.random.random_sample((EMBEDDINGS_DIM,))

    # Get only the words from the vocabulary
    words_set = set(words)
    wv = [vectors[words.index(lemma)] if lemma in words_set else unknown for lemma in vocabulary]

    print 'Known lemmas:', len(words_set.intersection(set(vocabulary))), '/', len(vocabulary)

    # Normalize each row (word vector) in the matrix to sum-up to 1
    row_norm = np.sum(np.abs(wv)**2, axis=-1)**(1./2)
    wv /= row_norm[:, np.newaxis]

    return wv


def unique(a):
    """

    :param a:
    :return:
    """
    indices = sorted(range(len(a)), key=a.__getitem__)
    indices = set(next(it) for k, it in
                  groupby(indices, key=a.__getitem__))
    return [x for i, x in enumerate(a) if i in indices]