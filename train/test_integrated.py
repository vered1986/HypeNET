import sys
sys.argv.insert(1, '--dynet-gpus')
sys.argv.insert(2, '1')
sys.argv.insert(3, '--dynet-mem')
sys.argv.insert(4, '8192')

sys.path.append('../common')

from lstm_common import *
from paths_lstm_classifier import *
from knowledge_resource import KnowledgeResource
from sklearn.metrics import precision_recall_fscore_support


EMBEDDINGS_DIM = 50


def main():
    """
    Load a pre-trained model of the LSTM-based integrated path-based and distributional method
    for hypernymy detection, and test it on the test set
    :return:
    """
    corpus_prefix = sys.argv[5]
    dataset_prefix = sys.argv[6]
    model_file_prefix = sys.argv[7]

    # Load the datasets
    print 'Loading the dataset...'
    test_set = load_dataset(dataset_prefix + 'test.tsv')
    y_test = [1 if 'True' in label else 0 for label in test_set.values()]

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = KnowledgeResource(corpus_prefix)
    print 'Done!'

    # Load the model
    classifier, lemma_index, pos_index, dep_index, dir_index = load_model(model_file_prefix)

    # Load the paths and create the feature vectors
    print 'Loading path files...'
    x_y_vectors_test, X_test = load_paths(corpus, test_set.keys(), lemma_index, pos_index, dep_index, dir_index)

    print 'Evaluation:'
    pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test)
    p, r, f1, support = precision_recall_fscore_support(y_test, pred, average='binary')
    print 'Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f1)

    # Write the predictions to a file
    relations = ['False', 'True']
    output_predictions(model_file_prefix + '.test_predictions', relations, pred, test_set.keys(), y_test)


def load_paths(corpus, dataset_keys, lemma_index, pos_index, dep_index, dir_index):
    """
    Override load_paths from lstm_common to include (x, y) vectors
    :param corpus:
    :param dataset_keys:
    :return:
    """
    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    paths_x_to_y = [{ vectorize_path(path, lemma_index, pos_index, dep_index, dir_index) : count
                      for path, count in get_paths(corpus, x_id, y_id).iteritems() }
                    for (x_id, y_id) in keys]
    paths_x_to_y = [ { p : c for p, c in paths_x_to_y[i].iteritems() if p is not None } for i in range(len(keys)) ]

    empty = [dataset_keys[i] for i, path_list in enumerate(paths_x_to_y) if len(path_list.keys()) == 0]
    print 'Pairs without paths:', len(empty), ', all dataset:', len(dataset_keys)

    # Get the word embeddings for x and y (get a lemma index)
    x_y_vectors = [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset_keys]

    return x_y_vectors, paths_x_to_y


if __name__ == '__main__':
    main()
