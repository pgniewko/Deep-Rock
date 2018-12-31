#! /bin/bash -x

LATTICE="/Users/pawel/Projects/Deep-Rock/output/LATTICE"
LB_PATH="/Users/pawel/Projects/Deep-Rock/output/LB"

for INDEX in `seq 1 1000`; do

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
