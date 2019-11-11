#! /bin/bash -x
# Create a list of files that are needed for the CNN input preparation script.
# In the following, we generate a list of files for L=64 and k=2.
# Refer to the paper for the details.

LATTICE="/Users/pawel/Projects/Deep-Rock/output/LATTICE"
LB_PATH="/Users/pawel/Projects/Deep-Rock/output/LB"

rm FILES_LIST.txt 2> /dev/null

for INDEX in `seq 1 10000`; do

FILE1=$LATTICE"/64_2_"$INDEX".bin.txt"
FILE2=$LATTICE"/64_2_"$INDEX".out"
FILE3=$LB_PATH"/64_2_"$INDEX".dat"

if [ -f $FILE1 ]
then
  if [ -f $FILE2 ]
  then
    if [ -f $FILE3 ]
    then
      echo $FILE1 $FILE2 $FILE3 >> FILES_LIST.txt
    fi
  fi
fi


done
