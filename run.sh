#! /bin/bash -x 

for INDEX in `seq 1 10000`; do
  SF="./src/lattice/mc.py"
  $SF 64 2 $INDEX
  $SF 64 4 $INDEX
done
