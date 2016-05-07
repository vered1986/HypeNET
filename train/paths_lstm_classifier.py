import numpy as np
import math

from pycnn import *
import pycnn
from sklearn.base import BaseEstimator

NUM_LAYERS = 2
HIDDEN_DIM = 60
LEMMA_DIM = 50
POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1


class PathLSTMClassifier(BaseEstimator):

    def __init__(self, num_lemmas, num_pos, num_dep, num_directions=5, n_epochs=10, num_relations=2,
                 alpha=0.01, lemma_embeddings=None, dropout=0.0, use_xy_embeddings=False):
        """"
        Initialize the LSTM
        :param num_lemmas Number of distinct lemmas
        :param num_pos Number of distinct part of speech tags
        :param num_dep Number of distinct depenedency labels
        :param num_directions Number of distinct path directions (e.g. >,<)
        :param n_epochs Number of training epochs
        :param num_relations Number of classes (e.g. binary = 2)
        :param alpha Learning rate
        :param lemma_embeddings Pre-trained word embedding vectors
        :param dropout Dropout rate
        :param use_xy_embeddings Whether to concatenate x and y word embeddings to the network input
        """
        self.n_epochs = n_epochs
        self.num_lemmas = num_lemmas
        self.num_pos = num_pos
        self.num_dep = num_dep
        self.num_directions = num_directions
        self.num_relations = num_relations
        self.alpha = alpha
        self.dropout = dropout
        self.use_xy_embeddings = use_xy_embeddings

        self.wv = None
        self.update = True
        if lemma_embeddings is not None:
            self.wv = lemma_embeddings

        # Create the network
        print 'Creating the network...'
        self.builder, self.model = create_computation_graph(self.num_lemmas, self.num_pos, self.num_dep,
                                                            self.num_directions, self.num_relations, self.wv,
                                                            self.use_xy_embeddings)
        print 'Done!'

    def fit(self, X_train, y_train, x_y_vectors=None):
        """
        Train the model
        """
        print 'Training the model...'
        train(self.builder, self.model, X_train, y_train, self.n_epochs, self.alpha, self.update,
              self.dropout, x_y_vectors)
        print 'Done!'

    def save_model(self, output_prefix):
        """
        Save the trained model to a file
        """
        self.model.save(output_prefix + '.model')

    def load_model(self, model_file_prefix):
        """
        Load the trained model from a file
        """
        self.model.load(model_file_prefix + '.model')

    def predict(self, X_test, x_y_vectors=None):
        """
        Predict the classification of the test set
        """
        model = self.model
        builder = self.builder

        test_pred = []

        # Predict every 100 instances together (memory consuming)
        for chunk in xrange(0, len(X_test), 100):
            renew_cg()
            path_cache = {}
            test_pred.extend([np.argmax(process_one_instance(
                builder, model, path_set, path_cache, False, dropout=0.0,
                x_y_vectors=x_y_vectors[chunk + i] if x_y_vectors is not None else None,
                num_hidden_layers=self.num_hidden_layers).npvalue())
                              for i, path_set in enumerate(X_test[chunk:chunk+100])])

        return test_pred

    def get_top_k_paths(self, all_paths, k=None, threshold=None):
        """
        Get the top k scoring paths
        """
        cg = renew_cg()
        path_scores = []
        lemma_lookup = self.model["lemma_lookup"]
        pos_lookup = self.model["pos_lookup"]
        dep_lookup = self.model["dep_lookup"]
        dir_lookup = self.model["dir_lookup"]
        builder = self.builder
        W = parameter(self.model["W"])

        for path in all_paths:
            path_embedding = get_path_embedding(builder, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path)

            if self.use_xy_embeddings:
                zero_word = pycnn._vecInputExpression(cg, [0.0] * LEMMA_DIM)
                path_embedding = concatenate([zero_word, path_embedding, zero_word])

            path_scores.append(softmax(W * path_embedding).npvalue()[1])

        path_scores = np.array(path_scores)
        indices = np.argsort(-path_scores)

        if k is not None:
            indices = indices[:k]

        top_paths = [(all_paths[index], path_scores[index]) for index in indices
                     if threshold is None or path_scores[index] >= threshold]
        return top_paths


def process_one_instance(builder, model, instance, path_cache, update=True, dropout=0.0, x_y_vectors=None):
    """
    Return the LSTM output vector of a single term-pair - the average path embedding
    :param builder: the LSTM builder
    :param model: the LSTM model
    :param update: whether to update the lemma embeddings
    :param instance: a Counter object with paths
    :return: the LSTM output vector of a single term-pair
    """

    W = parameter(model["W"])
    lemma_lookup = model["lemma_lookup"]
    pos_lookup = model["pos_lookup"]
    dep_lookup = model["dep_lookup"]
    dir_lookup = model["dir_lookup"]

    # Use the LSTM output vector and feed it to the MLP
    num_paths = reduce(lambda x, y: x + y, instance.itervalues())
    path_embbedings = [get_path_embedding_from_cache(path_cache, builder, lemma_lookup, pos_lookup, dep_lookup,
                                                     dir_lookup, path, update, dropout) * count
                       for path, count in instance.iteritems()]
    h = esum(path_embbedings) * (1.0 / num_paths)

    # Concatenate x and y embeddings
    if x_y_vectors is not None:
        x_vector, y_vector = lookup(lemma_lookup, x_y_vectors[0]), lookup(lemma_lookup, x_y_vectors[1])
        h = concatenate([x_vector, h, y_vector])

    return softmax(W * h)


def get_path_embedding_from_cache(cache, builder, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path,
                                  update=True, dropout=0.0):

    if path not in cache:
        cache[path] = get_path_embedding(builder, lemma_lookup, pos_lookup, dep_lookup,
                                         dir_lookup, path, update, dropout)
    return cache[path]


def get_path_embedding(builder, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path, update=True, drop=0.0):
    """
    Get a vector representing a path
    :param builder: the LSTM builder
    :param lemma_lookup: the lemma embeddings lookup table
    :param pos_lookup: the part-of-speech embeddings lookup table
    :param dep_lookup: the dependency label embeddings lookup table
    :param dir_lookup: the direction embeddings lookup table
    :param path: sequence of edges
    :param update: whether to update the lemma embeddings
    :return: a vector representing a path
    """

    # Concatenate the edge components to one vector
    inputs = [concatenate([word_dropout(lemma_lookup, edge[0], drop),
                           word_dropout(pos_lookup, edge[1], drop),
                           word_dropout(dep_lookup, edge[2], drop),
                           word_dropout(dir_lookup, edge[3], drop)])
                 for edge in path]

    return builder.initial_state().transduce(inputs)[-1]


def word_dropout(lookup_table, word, rate):
    """ Apply word dropout with dropout rate
    :param exp: expression vector
    :param rate: dropout rate
    :return:
    """
    new_word = np.random.choice([word, 0], size=1, p=[1 - rate, rate])[0]
    return lookup(lookup_table, new_word)


def train(builder, model, X_train, y_train, nepochs, alpha=0.01, update=True, dropout=0.0, x_y_vectors=None):
    """
    Train the LSTM
    :param builder: the LSTM builder
    :param model: LSTM RNN model
    :param X_train: the train instances
    :param y_train: the train labels
    :param nepochs: number of epochs
    :param alpha: the learning rate (only for SGD)
    :param dropout: dropout probability for all component embeddings
    :param update: whether to update the lemma embeddings
    """
    trainer = AdamTrainer(model, alpha=alpha)
    minibatch_size = min(10, len(y_train))
    nminibatches = int(math.ceil(len(y_train) / minibatch_size))

    for epoch in range(nepochs):

        total_loss = 0.0

        epoch_indices = np.random.permutation(len(y_train))

        for minibatch in range(nminibatches):

            path_cache = {}
            batch_indices = epoch_indices[minibatch * minibatch_size:(minibatch + 1) * minibatch_size]

            renew_cg()
            loss = esum([-log(pick(
                process_one_instance(builder, model, X_train[batch_indices[i]], path_cache, update, dropout,
                                     x_y_vectors=x_y_vectors[batch_indices[i]] if x_y_vectors is not None else None),
                y_train[batch_indices[i]])) for i in range(minibatch_size)])
            total_loss += loss.value() # forward computation
            loss.backward()
            trainer.update()

        trainer.update_epoch()
        total_loss /= len(y_train)
        print 'Epoch', (epoch + 1), '/', nepochs, 'Loss =', total_loss


def create_computation_graph(num_lemmas, num_pos, num_dep, num_directions, num_relations, wv=None,
                             use_xy_embeddings=False):
    """
    Initialize the model
    :param num_lemmas Number of distinct lemmas
    :param num_pos Number of distinct part of speech tags
    :param num_dep Number of distinct depenedency labels
    :param num_directions Number of distinct path directions (e.g. >,<)
    :param num_relations Number of classes (e.g. binary = 2)
    :param wv Pre-trained word embeddings file
    :param use_xy_embeddings Whether to concatenate x and y word embeddings to the network input
    :return:
    """
    model = Model()
    network_input = HIDDEN_DIM

    builder = LSTMBuilder(NUM_LAYERS, LEMMA_DIM + POS_DIM + DEP_DIM + DIR_DIM, network_input, model)

    # Concatenate x and y
    if use_xy_embeddings:
        network_input += 2 * LEMMA_DIM

    model.add_parameters("W", (num_relations, network_input))

    model.add_lookup_parameters("lemma_lookup", (num_lemmas, LEMMA_DIM))

    # Pre-trained word embeddings
    if wv is not None:
        model["lemma_lookup"].init_from_array(wv)

    model.add_lookup_parameters("pos_lookup", (num_pos, POS_DIM))
    model.add_lookup_parameters("dep_lookup", (num_dep, DEP_DIM))
    model.add_lookup_parameters("dir_lookup", (num_directions, DIR_DIM))

    return builder, model