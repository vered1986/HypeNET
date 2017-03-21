#!/usr/bin/env bash

wiki_dump_file=$1
resource_prefix=$2

# Parse wikipedia. Splitting to 20 files and running in parallel.
echo 'Parsing wikipedia...'
split -nl/20 $wiki_dump_file $wiki_dump_file"_";

for x in {a..t}
do
( python parse_wikipedia.py $wiki_dump_file"_a"$x $wiki_dump_file"_a"$x"_parsed" ) &
done
wait

triplet_file="wiki_parsed"
cat $wiki_dump_file"_a"*"_parsed" > $triplet_file

# Create the frequent paths file (take paths that occurred approximately at least 5 times. To take paths that occurred with at least 5 different pairs,
# replace with the commented lines - consumes much more memory).
# sort -u $triplet_file | cut -f3 -d$'\t' > paths;
# awk -F$'\t' '{a[$1]++; if (a[$1] == 5) print $1}' paths > frequent_paths;
# rm paths;
for x in {a..t}
do
( awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_a"$x"_parsed" > paths"_a"$x ) &
done
wait

cat paths_a* > paths_temp;
cat paths_temp | grep -v "$(printf '\t1$')" > frequent_paths_temp;
awk -F$'\t' '{i[$1]+=$2} END{for(x in i){print x"\t"i[x]}}' frequent_paths_temp > paths;
awk -F$'\t' '$2 >= 5 {print $1}' paths > frequent_paths;
rm paths_temp frequent_paths_temp paths_a*; # You can remove paths to save space, or keep it to change the threshold for frequent paths

# Create the terms file
awk -F$'\t' '{a[$1]++; if (a[$1] == 1) print $1}' wiki_parsed > left & PIDLEFT=$!
awk -F$'\t' '{a[$2]++; if (a[$2] == 1) print $2}' wiki_parsed > right & PIDRIGHT=$!
wait $PIDLEFT
wait $PIDRIGHT
cat left right | sort -u > terms;
rm left right &

# First step - create the term and path to ID dictionaries
echo 'Creating the resource from the triplets file...'
python create_resource_from_corpus_1.py frequent_paths terms $resource_prefix;

# Second step - convert the textual triplets to triplets of IDs. 
for x in {a..t}
do
( python create_resource_from_corpus_2.py "wiki_a"$x"_parsed" $resource_prefix ) &
done
wait

# Third step - use the ID-based triplet file and converts it to the '_l2r.db' file
for x in {a..t}
do
( awk -v OFS='\t' '{i[$0]++} END{for(x in i){print x, i[x]}}' "wiki_a"$x"_parsed_id" > id_triplet_file"_a"$x ) &
done
wait

cat id_triplet_file_* > id_triplet_file_temp;

for x in {0..4}
do
( gawk -F $'\t' '{ if($1%5==$x) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' id_triplet_file_temp > id_triplet_file_$x ) &
done
wait

cat id_triplet_file_* > id_triplet_file;
rm id_triplet_file_temp id_triplet_file_* $triplet_file"_"*;

python create_resource_from_corpus_3.py id_triplet_file $resource_prefix;

# You can delete triplet_file now and keep only id_triplet_file which is more efficient, or delete both.