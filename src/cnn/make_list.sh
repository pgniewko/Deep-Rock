#! /bin/bash -x

LATTICE="/Users/pawel/Projects/Deep-Rock/output/LATTICE"
LB_PATH="/Users/pawel/Projects/Deep-Rock/output/LB"

rm FILES_LIST.txt

for INDEX in `seq 1 10000`; do

FILE1=$LATTICE"/128_2_"$INDEX".bin.txt"
FILE2=$LATTICE"/128_2_"$INDEX".out"
FILE3=$LB_PATH"/128_2_"$INDEX".dat"

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
