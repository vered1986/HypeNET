import codecs
import spacy

from spacy.en import English
from docopt import docopt


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Parse the Wikipedia dump and create a triplets file, each line is formatted as follows: X\t\Y\tpath

    Usage:
        parse_wikipedia.py <in_file> <out_file>

        <in_file> = the Wikipedia dump file
        <out_file> = the output (parsed) file
    """)

    nlp = English()

    in_file = args['<in_file>']
    out_file = args['<out_file>']

    with codecs.open(in_file, 'r', 'utf-8') as f_in:
        with codecs.open(out_file, 'w', 'utf-8') as f_out:

            # Read the next paragraph
            for paragraph in f_in:

                # Skip empty lines
                paragraph = paragraph.replace("'''", '').strip()
                if len(paragraph) == 0:
                    continue

                parsed_par = nlp(unicode(paragraph))

                # Parse each sentence separately
                for sent in parsed_par.sents:
                    dependency_paths = parse_sentence(sent)
                    if len(dependency_paths) > 0:
                        print >> f_out, '\n'.join(['\t'.join(path) for path in dependency_paths])


def parse_sentence(sent):
    """
    Get all the dependency paths between nouns in the sentence
    :param sent: the sentence to parse
    :return: the list of entities and paths
    """

    # Get all noun indices
    indices = [(token, i, i) for i, token in enumerate(sent) if token.tag_[:2] == 'NN' and len(token.string.strip()) > 2]

    # Add noun chunks for the current sentence
    # Don't include noun chunks with only one word - these are nouns already included
    indices.extend([(np, np.start, np.end) for np in sent.doc.noun_chunks if sent.start <= np.start < np.end - 1 < sent.end])

    # Get all dependency paths between nouns, up to length 4
    term_pairs = [(x[0], y[0]) for x in indices for y in indices if x[2] < y[1]]
    paths = [path for path in map(shortest_path, term_pairs) if path is not None]
    paths = [p for path in paths for p in get_satellite_links(path)]
    paths = [path for path in map(clean_path, paths) if path is not None]

    return paths


def shortest_path((x, y)):
    """ Returns the shortest dependency path from x to y
    :param x: x token
    :param y: y token
    :return: the shortest dependency path from x to y
    """

    x_token = x
    y_token = y
    if not isinstance(x_token, spacy.tokens.token.Token):
        x_token = x_token.root
    if not isinstance(y_token, spacy.tokens.token.Token):
        y_token = y_token.root

    # Get the path from the root to each of the tokens
    hx = heads(x_token)
    hy = heads(y_token)

    # Get the lowest common head
    i = -1
    for i in xrange(min(len(hx), len(hy))):
        if hx[i] is not hy[i]:
            break

    if i == -1:
        lch_idx = 0
        if len(hy) > 0:
            lch = hy[lch_idx]
        elif len(hx) > 0:
            lch = hx[lch_idx]
        else:
            lch = None
    elif hx[i] == hy[i]:
        lch_idx = i
        lch = hx[lch_idx]
    else:
        lch_idx = i-1
        lch = hx[lch_idx]

    # The path from x to the lowest common head
    hx = hx[lch_idx+1:]
    if lch and check_direction(lch, hx, lambda h: h.lefts):
        return None
    hx = hx[::-1]

    # The path from the lowest common head to y
    hy = hy[lch_idx+1:]
    if lch and check_direction(lch, hy, lambda h: h.rights):
        return None

    return (x, hx, lch, hy, y)


def heads(token):
    """
    Return the heads of a token, from the root down to immediate head
    :param token:
    :return:
    """
    t = token
    hs = []
    while t is not t.head:
        t = t.head
        hs.append(t)
    return hs[::-1]


def check_direction(lch, hs, f_dir):
    """
    Make sure that the path between the term and the lowest common head is in a certain direction
    :param lch: the lowest common head
    :param hs: the path from the lowest common head to the term
    :param f_dir: function of direction
    :return:
    """
    return any(modifier not in f_dir(head) for head, modifier in zip([lch] + hs[:-1], hs))


def get_satellite_links((x, hx, lch, hy, y)):
    """
    Add the "satellites" - single links not already contained in the dependency path added on either side of each noun
    :param x: the X token
    :param y: the Y token
    :param hx: X's head tokens
    :param hy: Y's head tokens
    :param lch: the lowest common ancestor of X and Y
    :return: more paths, with satellite links
    """
    paths = [(None, x, hx, lch, hy, y, None)]

    x_lefts = [tok for tok in x.lefts]
    if len(x_lefts) > 0 and x_lefts[0].tag_ != 'PUNCT' and len(x_lefts[0].string.strip()) > 1:
        paths.append((x_lefts[0], x, hx, lch, hy, y, None))

    y_rights = [tok for tok in y.rights]
    if len(y_rights) > 0 and y_rights[0].tag_ != 'PUNCT' and len(y_rights[0].string.strip()) > 1:
        paths.append((None, x, hx, lch, hy, y, y_rights[0]))

    return paths


def edge_to_string(token, is_head=False):
    """
    Converts the token to an edge string representation
    :param token: the token
    :return: the edge string
    """
    t = token
    if not isinstance(token, spacy.tokens.token.Token):
        t = token.root

    return '/'.join([token_to_lemma(token), t.pos_, t.dep_ if t.dep_ != '' and not is_head else 'ROOT'])


def argument_to_string(token, edge_name):
    """
    Converts the argument token (X or Y) to an edge string representation
    :param token: the X or Y token
    :param edge_name: 'X' or 'Y'
    :return:
    """
    if not isinstance(token, spacy.tokens.token.Token):
        token = token.root

    return '/'.join([edge_name, token.pos_, token.dep_ if token.dep_ != '' else 'ROOT'])


def direction(dir):
    """
    Print the direction of the edge
    :param dir: the direction
    :return: a string representation of the direction
    """
    # Up to the head
    if dir == UP:
        return '>'
    # Down from the head
    elif dir == DOWN:
        return '<'


def token_to_string(token):
    """
    Convert the token to string representation
    :param token:
    :return:
    """
    if not isinstance(token, spacy.tokens.token.Token):
        return ' '.join([t.string.strip().lower() for t in token])
    else:
        return token.string.strip().lower()


def token_to_lemma(token):
    """
    Convert the token to string representation
    :param token: the token
    :return: string representation of the token
    """
    if not isinstance(token, spacy.tokens.token.Token):
        return token_to_string(token)
    else:
        return token.lemma_.strip().lower()


def clean_path((set_x, x, hx, lch, hy, y, set_y)):
    """
    Filter out long paths and pretty print the short ones
    :return: the string representation of the path
    """
    set_path_x = []
    set_path_y = []
    lch_lst = []

    if set_x:
        set_path_x = [edge_to_string(set_x) + direction(DOWN)]
    if set_y:
        set_path_y = [direction(UP) + edge_to_string(set_y)]

    # X is the head
    if isinstance(x, spacy.tokens.token.Token) and lch == x:
        dir_x = ''
        dir_y = direction(DOWN)
    # Y is the head
    elif isinstance(y, spacy.tokens.token.Token) and lch == y:
        dir_x = direction(UP)
        dir_y = ''
    # X and Y are not heads
    else:
        lch_lst = [edge_to_string(lch, is_head=True)] if lch else []
        dir_x = direction(UP)
        dir_y = direction(DOWN)

    len_path = len(hx) + len(hy) + len(set_path_x) + len(set_path_y) + len(lch_lst)

    if len_path <= MAX_PATH_LEN:
        cleaned_path = '_'.join(set_path_x + [argument_to_string(x, 'X') + dir_x] +
                                [edge_to_string(token) + direction(UP) for token in hx] +
                                lch_lst +
                                [direction(DOWN) + edge_to_string(token) for token in hy] +
                                [dir_y + argument_to_string(y, 'Y')] + set_path_y)
        return token_to_string(x), token_to_string(y), cleaned_path
    else:
        return None


# Constants
MAX_PATH_LEN = 4
UP = 1
DOWN = 2

if __name__ == '__main__':
    main()