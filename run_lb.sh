#! /bin/bash -x 

LB_PATH="/Users/pawel/Desktop/LB/"
SF="./src/lb/porous-2d"

for INDEX in `seq 1 10000`; do

  FILE="/Users/pawel/Projects/Deep-Rock/output/LATTICE/64_2_"$INDEX".lattice"
  if [ -f "$FILE" ]
  then
    $SF 192 64 $FILE 64 2 $INDEX $LB_PATH
  else
    echo 0.0 0.0 > $LB_PATH"/64_2_"$INDEX".dat"
  fi

  FILE="/Users/pawel/Projects/Deep-Rock/output/LATTICE/64_4_"$INDEX".lattice"
  if [ -f "$FILE" ]
  then
    $SF 192 64 $FILE 64 4 $INDEX $LB_PATH
  else
    echo 0.0 0.0 > $LB_PATH"/64_4_"$INDEX".dat"
  fi

done
