import bsddb
import codecs

from docopt import docopt


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Creates a knowledge resource from triplets file. Third step, uses the ID-based triplet file
    and converts it to the '_l2r.db' file.

    Usage:
        create_resource_from_corpus_3.py <id_triplet_file> <resource_prefix>

        <id_triplet_file> = a file containing the int triplets, formated as X_id\tY_id\tpath_id\tcount, where
        count is the number of times X and Y occurred together in this path. You can obtain such a file by
        counting the number of occurrences of each line in the file produced by the second step, e.g.:
        awk '{i[$0]++} END{for(x in i){print x"\t"i[x]}}' triplet_file > id_triplet_file

        If you split the files in the second step, apply this command to each one of them, and then sum them up, e.g.:
        for each i, run: awk '{i[$0]++} END{for(x in i){print x"\t"i[x]}}' triplet_file_i > id_triplet_file_i
        cat id_triplet_file_* > id_triplet_file_temp

        Then, run: awk -F$'\t' '{i[$1,"\t",$2,"\t",$3]+=$4} END{for(x in i){print x"\t"i[x]}}' id_triplet_file_temp > id_triplet_file

        <resource_prefix> = the file names' prefix for the resource files
    """)

    id_triplet_file = args['<id_triplet_file>']
    resource_prefix = args['<resource_prefix>']

    l2r_db = bsddb.btopen(resource_prefix + '_l2r.db', 'c')

    with codecs.open(id_triplet_file) as f_in:
        for line in f_in:
            try:
                x, y, path, count = line.strip().split('\t')
            except:
                print line
                continue

            key = str(x) + '###' + str(y)
            current = '%s:%s' % (path, count)

            if key in l2r_db:
                current = l2r_db[key] + ',' + current

            l2r_db[key] = current

    l2r_db.sync()


if __name__ == '__main__':
    main()
