import sys
sys.argv.insert(1, '--cnn-mem')
sys.argv.insert(2, '8192')

from sklearn.metrics import precision_recall_fscore_support
from paths_lstm_classifier import PathLSTMClassifier
from lstm_common import *
from knowledge_resource import KnowledgeResource
from collections import defaultdict
from itertools import count

EMBEDDINGS_DIM = 50


def main():

    # The LSTM-based integrated pattern-based and distributional method for hypernymy detection
    corpus_prefix = sys.argv[3]
    dataset_prefix = sys.argv[4]
    output_file = sys.argv[5]
    embeddings_file = sys.argv[6]
    alpha = float(sys.argv[7])
    word_dropout_rate = float(sys.argv[8])

    np.random.seed(133)
    relations = ['False', 'True']

    # Load the datasets
    print 'Loading the dataset...'
    train_set = load_dataset(dataset_prefix + 'train.tsv')
    test_set = load_dataset(dataset_prefix + 'test.tsv')
    val_set = load_dataset(dataset_prefix + 'val.tsv')
    y_train = [1 if 'True' in train_set[key] else 0 for key in train_set.keys()]
    y_test = [1 if 'True' in test_set[key] else 0 for key in test_set.keys()]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # y_val = [1 if 'True' in val_set[key] else 0 for key in val_set.keys()]
    dataset_keys = train_set.keys() + test_set.keys() + val_set.keys()
    print 'Done!'

    # Load the paths and create the feature vectors
    print 'Loading path files...'
    x_y_vectors, dataset_instances, lemma_index, pos_index, dep_index, dir_index, lemma_inverted_index, \
        pos_inverted_index, dep_inverted_index, dir_inverted_index = load_paths(corpus_prefix, dataset_keys)
    print 'Done!'
    print 'Number of lemmas %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
          (len(lemma_index), len(pos_index), len(dep_index), len(dir_index))

    # Load the word embeddings
    print 'Initializing word embeddings...'
    if embeddings_file is not None:
        wv = load_embeddings(embeddings_file, lemma_index.keys())

    X_train = dataset_instances[:len(train_set)]
    X_test = dataset_instances[len(train_set):len(train_set)+len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # X_val = dataset_instances[len(train_set)+len(test_set):]

    x_y_vectors_train = x_y_vectors[:len(train_set)]
    x_y_vectors_test = x_y_vectors[len(train_set):len(train_set)+len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # x_y_vectors_val = x_y_vectors[len(train_set)+len(test_set):]

    # Create the classifier
    classifier = PathLSTMClassifier(num_lemmas=len(lemma_index), num_pos=len(pos_index),
                                    num_dep=len(dep_index),num_directions=len(dir_index), n_epochs=3,
                                    num_relations=2, lemma_embeddings=wv, dropout=word_dropout_rate, alpha=alpha,
                                    use_xy_embeddings=True)

    # print 'Training with regularization = %f, learning rate = %f, dropout = %f...' % (reg, alpha, dropout)
    print 'Training with learning rate = %f, dropout = %f...' % (alpha, word_dropout_rate)
    classifier.fit(X_train, y_train, x_y_vectors=x_y_vectors_train)

    print 'Evaluation:'
    pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test)
    p, r, f1, support = precision_recall_fscore_support(y_test, pred, average='binary')
    print 'Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f1)

    # Save the best model to a file
    classifier.save_model(output_file)

    # Write the predictions to a file
    predictions = [1 if y > 0.5 else 0 for y in pred]
    output_predictions(output_file + '.predictions', relations, predictions, test_set.keys(), y_test)

    # Retrieve k-best scoring paths
    all_paths = unique([path for path_list in dataset_instances for path in path_list])
    top_k = classifier.get_top_k_paths(all_paths, 1000)

    with codecs.open(output_file + '.paths', 'w', 'utf-8') as f_out:
        for path, score in top_k:
            path_str = '_'.join([reconstruct_edge(edge, lemma_inverted_index, pos_inverted_index,
                                                  dep_inverted_index, dir_inverted_index) for edge in path])
            print >> f_out, '\t'.join([path_str, str(score)])


def load_paths(corpus_prefix, dataset_keys):
    """
    Override load_paths from lstm_common to include (x, y) vectors
    :param corpus_prefix:
    :param dataset_keys:
    :return:
    """

    # Define the dictionaries
    lemma_index = defaultdict(count(0).next)
    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    dummy = lemma_index['#NOPATH#']
    dummy = pos_index['#NOPATH#']
    dummy = dep_index['#NOPATH#']
    dummy = dir_index['#NOPATH#']

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = KnowledgeResource(corpus_prefix)
    print 'Done!'

    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    paths_x_to_y = [{ vectorize_path(path, lemma_index, pos_index, dep_index, dir_index) : count
                      for path, count in get_paths(corpus, x_id, y_id).iteritems() }
                    for (x_id, y_id) in keys]
    paths_x_to_y = [ { p : c for p, c in paths_x_to_y[i].iteritems() if p is not None } for i in range(len(keys)) ]

    paths = paths_x_to_y

    empty = [dataset_keys[i] for i, path_list in enumerate(paths) if len(path_list.keys()) == 0]
    print 'Pairs without paths:', len(empty), ', all dataset:', len(dataset_keys)

    # Get the word embeddings for x and y (get a lemma index)
    x_y_vectors = [(lemma_index[x], lemma_index[y]) for (x, y) in dataset_keys]

    lemma_inverted_index = { i : p for p, i in lemma_index.iteritems() }
    pos_inverted_index = { i : p for p, i in pos_index.iteritems() }
    dep_inverted_index = { i : p for p, i in dep_index.iteritems() }
    dir_inverted_index = { i : p for p, i in dir_index.iteritems() }

    return x_y_vectors, paths, lemma_index, pos_index, dep_index, dir_index, \
           lemma_inverted_index, pos_inverted_index, dep_inverted_index, dir_inverted_index


if __name__ == '__main__':
    main()
