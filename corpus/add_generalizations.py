import codecs

from docopt import docopt
from generalizations_common import generalize_path
from collections import defaultdict
from itertools import count


def main():

    args = docopt("""Add the generalized paths to the resource.

        Usage:
            add_generalizations.py <in_prefix>

            <in_prefix> = the directory and prefix for the resource files (e.g. /home/user/resources/res1).
        """)

    in_prefix = args['<in_prefix>']
    out_prefix = in_prefix + '_gen'

    # Load the paths
    with codecs.open(in_prefix + 'Paths.txt', 'r', 'utf-8') as f_in:
        lines = [tuple(line.strip().split('\t')) for line in f_in]
        orig_path_id_to_label = { int(path_id) : path for (path_id, path) in lines }
        path_label_to_id = defaultdict(count(0).next)
        path_id_to_label = {}

    with codecs.open(in_prefix + '-l2r.txt', 'r', 'utf-8') as f_in:
        with codecs.open(out_prefix + '-l2r.txt', 'w', 'utf-8') as f_l2r_out:

            for line in f_in:

                x_id, y_id, path_id = map(int, line.strip().split('\t'))
                curr_path = orig_path_id_to_label[path_id]
                curr_path_id = path_label_to_id[curr_path]
                path_id_to_label[curr_path_id] = curr_path
                print >> f_l2r_out, '\t'.join(map(str, [x_id, y_id, curr_path_id]))

                for new_path in generalize_path(curr_path):
                    new_path_id = path_label_to_id[new_path]
                    path_id_to_label[new_path_id] = new_path
                    print >> f_l2r_out, '\t'.join(map(str, [x_id, y_id, new_path_id]))

    with codecs.open(out_prefix + 'Paths.txt', 'w', 'utf-8') as f_paths_out:
        for (path_id, path) in path_id_to_label.iteritems():
            print >> f_paths_out, '\t'.join(map(str, (path_id, path)))


if __name__ == '__main__':
    main()
