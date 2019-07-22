#! /usr/bin/env python
#
# usage:
# ./prepare_data.py FILES_LIST.txt ../../output/CNN/images.txt  ../../output/CNN/values.txt

import sys
import numpy as np

fin = open(sys.argv[1], 'rU')

fout1 = open(sys.argv[2], 'w')
fout2 = open(sys.argv[3], 'w')

for line in fin:
    pairs = line.split()
    file_1 = pairs[0]
    file_2 = pairs[1]
    file_3 = pairs[2]

    latt = np.loadtxt(file_1, dtype=np.dtype(int))
    Lx, Ly = latt.shape

    s = ""
    for i in range(Lx - 1, -1, -1):
        for j in range(Ly - 1, -1, -1):
            s += str(latt[i][j]) + " "
    s += "\n"
    fout1.write(s)

    vals2 = np.loadtxt(file_2, dtype=np.dtype(float))
    vals3 = np.loadtxt(file_3, dtype=np.dtype(float))

    porosity = vals2[0]
    perc = vals2[1]
    permea = vals3[0]
    tau = vals3[1]
    fout2.write("{} {} {} {}\n".format(porosity, perc, permea, tau))
