#! /bin/bash -x 
#

OUTPUT_PATH="/Users/pawel/Desktop/LATTICE/"
PERCO="./src/lattice/mc.py"

for L in 64 128; do
  for k in 2 4; do
    for INDEX in `seq 1 10000`; do
      $PERCO $L $k $INDEX $OUTPUT_PATH
    done
  done
done
