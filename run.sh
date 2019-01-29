#! /bin/bash -x 

OPATH="/Users/pawel/Desktop/LATTICE/"
SF="./src/lattice/mc.py"

for L in 64 128; do
  for k in 2 4; do
    for INDEX in `seq 1 10000`; do
      $SF $L $k $INDEX $OPATH
    done
  done
done
