# HypeNET: Integrated Path-based and Distributional Method for Hypernymy Detection

This is the code used in the paper:

<b>"Improving Hypernymy Detection with an Integrated Path-based and Distributional Method"</b><br/>
Vered Shwartz, Yoav Goldberg and Ido Dagan. ACL 2016.

It is used to classify hypernymy relations between term-pairs, using disributional information on each term, and path-based information, encoded using an LSTM.

***

The repository contains the following directories:
* common - the knowledge resource class, which is used by other models to save the path data from the corpus (should be copied to other directories).
* corpus - code for parsing the corpus and extracting paths.
* dataset - code for creating the dataset used in the paper, and the dataset itself.
* lstm - code for training and testing both variants of our model (path-based and integrated).

Detailed Instructions: TBD