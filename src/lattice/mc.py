#! /usr/bin/env python

import sys
import numpy as np
from scipy.ndimage import measurements, label, generate_binary_structure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


def get_lattice(L, k, phi):
    lattice = np.ones([L,L])
    area = L * L
    volume_fraction = 1.0 - np.sum( np.sum(lattice) ) / area
    
    while volume_fraction < phi:
        volume_fraction = 1.0 - np.sum( np.sum(lattice) ) / area
        i, j = np.random.randint(L,size=2)
        for m in range(k):
            for n in range(k):
                lattice[(i+m) % L][(j+n) % L] = 0

    return lattice


def process_pbc(labeled_lattice, L):
    edge_sites = []
    for i in range(L):
        edge_sites.append( [0, i] )
    for i in range(L):
        edge_sites.append( [L-1, i] )
       
    num = len(edge_sites)

    i = 0
    while i < num:
        xi, yi = edge_sites[i]
        
        i += 1

        label_i = labeled_lattice[xi][yi]
        if xi == 0:
            label_j = labeled_lattice[L-1][yi]
            if label_i != 0 and label_j != 0 and label_i != label_j:
                mask = labeled_lattice == label_j
                labeled_lattice[mask] = label_i
                i = 0
                continue
        
        if xi == L-1:
            label_j = labeled_lattice[0][yi]
            if label_i != 0 and label_j != 0 and label_i != label_j:
                mask = labeled_lattice == label_j
                labeled_lattice[mask] = label_i
                i = 0
                continue
            
        
    return labeled_lattice


def find_percolating_cluster(labeled_lattice, L):
    left = []
    right = []
    percolating_ids = []

    for i in range(L):
        if labeled_lattice[i][0] != 0:
            left.append( labeled_lattice[i][0] )
        
        if labeled_lattice[i][L-1] != 0:
            right.append( labeled_lattice[i][L-1] )
        
    for el_1 in left:
        for el_2 in right:
            if el_1 == el_2:
                percolating_ids.append( el_2 )


    return list(set(percolating_ids))


def save_lattice(labeled_lattice_, L, k, seed, perc_ids, phi, path_="./LATTICE/"):
    labeled_lattice = labeled_lattice_.copy()
    lattice_out = np.zeros( [L,3*L] )
    for i in range(L):
        for j in range(L):
            if labeled_lattice_[i][j] == 0:
                labeled_lattice[i][j] = 0
                lattice_out[i][j + L] = 1
            elif labeled_lattice_[i][j] in perc_ids:
                labeled_lattice[i][j] = 1
                lattice_out[i][j + L] = 0
            else:
                labeled_lattice[i][j] = 2
                lattice_out[i][j + L] = 1

    colors = ['yellow','blue','green']  # R -> G -> B
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
    fig, ax = plt.subplots()
    i = ax.imshow(np.flip(labeled_lattice, axis=1), cmap=cm, interpolation='nearest')

    plot_out = path_ + str(L) + "_" + str(k) + "_" + str(seed) + ".png"
    plt.savefig(plot_out)
    
    s = ""
    for j in range(3*L-1,-1,-1):
        for i in range(L-1,-1,-1):
            s += str( int(lattice_out[i][j]) ) + " "

    if len(perc_ids) >  0:
        nout = path_ + str(L) + "_" + str(k) + "_" + str(seed) + ".lattice"
        fo = open(nout, 'w')
        fo.write(s)
        fo.close()


    nout = path_ + str(L) + "_" + str(k) + "_" + str(seed) + ".out"
    fo = open(nout, 'w')
    if len(perc_ids) >  0:
        fo.write( str(phi) + " 1 \n")
    else:
        fo.write( str(phi) + " 0 \n")
    fo.close()


    nout = path_ + str(L) + "_" + str(k) + "_" + str(seed) + ".bin.txt"
    fo = open(nout, 'w')
    for i in range(L):
        s = ""
        for j in range(L):
            if labeled_lattice_[i][j] == 0:
                s += str(0) + " " 
            
            else:
                s += str(1) + " " 

        s += "\n"
        fo.write( s )

    fo.close()


if __name__ == "__main__":

    L = int(sys.argv[1])
    k = int(sys.argv[2])
    seed_ = int(sys.argv[3])
    np.random.seed( seed_ )
    
    phi = np.random.uniform(0, 1)
    s = generate_binary_structure(2,1)
    
    
    lattice_2d = get_lattice(L, k,  phi)
    latt_1 = lattice_2d.copy()
    latt_2 = np.rot90( lattice_2d.copy() )

    # PROCESS LATT 1
    labeled_array, _ = label(latt_1, structure=s)
    labeled_array = process_pbc(labeled_array, L)
    perc_ids = find_percolating_cluster(labeled_array, L)
    if len(perc_ids)  > 0:
        save_lattice(labeled_array, L, k, seed_, perc_ids, 1 - phi, path_="/Users/pawel/Desktop/LATTICE/")
    else:
        # PROCESS LATT 2
        labeled_array, _ = label(latt_2, structure=s)
        labeled_array = process_pbc(labeled_array, L)
        perc_ids = find_percolating_cluster(labeled_array, L)
        save_lattice(labeled_array, L, k, seed_, perc_ids, 1 - phi, path_="/Users/pawel/Desktop/LATTICE/")

