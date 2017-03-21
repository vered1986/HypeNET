import bsddb
import codecs

from docopt import docopt


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Creates a knowledge resource from triplets file. Second step, uses the resource files
    already created and converts the textual triplet file to a triplet file with IDs.

    Usage:
        create_resource_from_corpus_2.py <triplet_file> <resource_prefix>

        <triplet_file> = a file containing the text triplets, formated as X\tY\tpath.
        You can run this script on multiple portions of the triplet file at once and concatenate the output.
        <resource_prefix> = the file names' prefix for the resource files
    """)

    triplet_file = args['<triplet_file>']
    resource_prefix = args['<resource_prefix>']

    # Load the resource DBs
    term_to_id_db = bsddb.btopen(resource_prefix + '_term_to_id.db')
    path_to_id_db = bsddb.btopen(resource_prefix + '_path_to_id.db')

    with codecs.open(triplet_file) as f_in:
        with codecs.open(triplet_file + '_id', 'w') as f_out:
            for line in f_in:
                try:
                    x, y, path = line.strip().split('\t')
                except:
                    print line
                    continue

                # Frequent path
                x_id, y_id, path_id = term_to_id_db[x], term_to_id_db[y], path_to_id_db.get(path, -1)
                if path_id != -1:
                    print >> f_out, '\t'.join(map(str, (x_id, y_id, path_id)))


if __name__ == '__main__':
    main()
