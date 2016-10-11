# HypeNET: Integrated Path-based and Distributional Method for Hypernymy Detection

This is the code used in the paper:

<b>"Improving Hypernymy Detection with an Integrated Path-based and Distributional Method"</b><br/>
Vered Shwartz, Yoav Goldberg and Ido Dagan. ACL 2016. [link](http://arxiv.org/abs/1603.06076)

It is used to classify hypernymy relations between term-pairs, using disributional information on each term, and path-based information, encoded using an LSTM.

***

Consider using our new project, [LexNET](https://github.com/vered1986/LexNET)! It supports classification of multiple semantic relations, and contains several model enhancements and detailed documentation.

***

<b>Prerequisites:</b>
* Python 2.7
* Numpy
* scikit-learn
* [bsddb](https://docs.python.org/2/library/bsddb.html)
* [PyCNN](https://github.com/clab/cnn/)

<b>Quick Start:</b>

The repository contains the following directories:
* common - the knowledge resource class, which is used by other models to save the path data from the corpus (should be copied to other directories).
* corpus - code for parsing the corpus and extracting paths, including the generalizations made for the baseline method.
* dataset - code for creating the dataset used in the paper, and the dataset itself.
* train - code for training and testing both variants of our model (path-based and integrated).

To train the integrated model, run:

`train_integrated.py [corpus_prefix] [dataset_prefix] [model_prefix_file] [embeddings_file] [alpha] [word_dropout_rate]`

Where:
* `corpus_prefix` is the file path and prefix of the corpus files, e.g. `corpus/wiki`, such that the directory corpus contains the `wiki_*.db` files created by `create_resource_from_corpus.py`.
* `dataset_prefix` is the file path of the dataset files, e.g. `dataset/rnd`, such that this directory contains 3 files: `train.tsv`, `test.tsv` and `val.tsv`.
* `model_prefix_file` is the output directory and prefix for the model files. The model is saved in 3 files: `.model`, `.params` and `.dict.`
In addition, the test set predictions are saved in `.predictions`, and the prominent paths are saved to `.paths`.
* `embeddings_file` is the pre-trained word embeddings file, in txt format (i.e., every line consists of the word, followed by a space, and its vector. See [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) for an example.)
* `alpha` is the learning rate (default=0.001).
* `word_dropout_rate` is the... word dropout rate.

Similarly, you can train the path-based model with `train_path_based.py` or test any of these pre-trained models using `test_integrated.py` and `test_path_based.py` respectively.
