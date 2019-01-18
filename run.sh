#! /bin/bash -x 

SF="./src/lattice/mc.py"

for L in 64 128 258; do
  for k in 2; do
    for INDEX in `seq 1 10000`; do
      $SF $L $k $INDEX
    done
  done
done
