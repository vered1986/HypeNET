import codecs

from collections import defaultdict
from itertools import count
from docopt import docopt


def main():
    """
    Split the dataset lexically
    Originally based on: Do Supervised Distributional Methods Really Learn Lexical Inference Relations?
    Omer Levy, Steffen Remus, Chris Biemann, and Ido Dagan. NAACL 2015.
    """

    # Get the arguments
    args = docopt("""Split the dataset lexically

    Usage:
        split_dataset_lexically.py <dataset_file> <output_directory>

        <dataset_file> = the original dataset file
        <output_directory> = the directory in which the train/test/validation sets should be saved
    """)

    dataset = args['<dataset_file>']
    output_directory = args['<output_directory>']

    # Read the dataset and assign an index for each word
    vocab = set()
    dict = defaultdict(count(0).next)
    data = []

    with codecs.open(dataset, 'r', 'utf-8') as f_in:
        for line in f_in:
            x, y, c = line.strip().split('\t')
            data.append((x, y, c))
            vocab.add(x)
            vocab.add(y)

    [dict[w] for w in vocab]

    # Split to train, test and validation such that each set contains a distinct vocabulary
    lex = [[], [], []]
    lex_mixed = []

    for x, y, c in data:
        hash_x = hash_word(dict[x])
        hash_y = hash_word(dict[y])
        if hash_x == hash_y:
            lex[hash_x].append((x, y, c))
        else:
            lex_mixed.append((x, y, c))

    lex_train, lex_test, lex_val = lex

    print 'Train:', len(lex_train), ', Test:', len(lex_test), ', Val:', len(lex_val)

    with codecs.open(output_directory + '/train.tsv', 'w', 'utf-8') as f_out:
        for x, y, c in lex_train:
            print >> f_out, '\t'.join([x, y, c])
            
    with codecs.open(output_directory + '/test.tsv', 'w', 'utf-8') as f_out:
        for x, y, c in lex_test:
            print >> f_out, '\t'.join([x, y, c])
            
    with codecs.open(output_directory + '/val.tsv', 'w', 'utf-8') as f_out:
        for x, y, c in lex_val:
            print >> f_out, '\t'.join([x, y, c])


def hash_word(s):
    """
    Decide for which set to assign this word, such that most words will
    be assigned to the train set, then to the test set and the least words will
    be assigned to the validation set
    :param s:
    :return:
    """
    f = s % 14
    if f < 7:
        return 0
    elif f < 11:
        return 1
    else:
        return 2


if __name__ == '__main__':
    main()