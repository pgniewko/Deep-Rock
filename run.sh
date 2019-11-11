#! /bin/bash -x 
#
# The path to the directory where geneated binary packings are saved
OUTPUT_PATH="/Users/pawel/Desktop/LATTICE/"
# The path to the packing generation code
PERCO="./src/lattice/mc.py"

# L - lattice size (number of lattice units)
# k - square ('rock') size (in lattice units)
# 10.000 random packings are going to be generated (phi=[0, 1])
for L in 64 128; do
  for k in 2 4; do
    for INDEX in `seq 1 10000`; do
      $PERCO $L $k $INDEX $OUTPUT_PATH
    done
  done
done
