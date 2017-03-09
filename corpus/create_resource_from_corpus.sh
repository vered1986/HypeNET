#!/usr/bin/env bash

triplet_file=$1
resource_prefix=$2

# Create the frequent paths file
sort -u $triplet_file | cut -f3 -d$'\t' > paths;
awk -F$'\t' '{a[$1]++; if (a[$1] == 5) print $1}' paths > frequent_paths;
rm paths;

# Create the terms file
awk -F$'\t' '{a[$1]++; if (a[$1] == 1) print $1}' wiki_parsed > left & PIDLEFT=$!
awk -F$'\t' '{a[$2]++; if (a[$2] == 1) print $2}' wiki_parsed > right & PIDRIGHT=$!
wait $PIDLEFT
wait $PIDRIGHT
cat left right | sort -u > terms;
rm left right &

# First step - create the term and path to ID dictionaries
python create_resource_from_corpus_1.py frequent_paths terms $resource_prefix;

# Second step - convert the textual triplets to triplets of IDs. Splitting to 20 files and running in parallel.
split -nl/20 $triplet_file $triplet_file"_";

for x in {a..t}
do
( python create_resource_from_corpus_2.py $triplet_file"_a"$x $resource_prefix ) &
done
wait

# Third step - use the ID-based triplet file and converts it to the '_l2r.db' file

for x in {a..t}
do
( awk -v OFS='\t' '{i[$0]++} END{for(x in i){print x, i[x]}}' $triplet_file"_a"$x"_id" > id_triplet_file"_a"$x ) &
done
wait

cat id_triplet_file_* > id_triplet_file_temp;
awk -F$'\t' '{i[$1,"\t",$2,"\t",$3]+=$4} END{for(x in i){print x"\t"i[x]}}' id_triplet_file_temp > id_triplet_file;

rm id_triplet_file_temp id_triplet_file_* $triplet_file"_"*;

python create_resource_from_corpus_3.py id_triplet_file $resource_prefix;

# You can delete triplet_file now and keep only id_triplet_file which is more efficient, or delete both.