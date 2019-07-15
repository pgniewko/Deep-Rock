>Notice: This is research code that will not necessarily be maintained in the future.
>The code is under development so make sure you are using the most recent version.
>We welcome bug reports and PRs but make no guarantees about fixes or responses.

DESCRIPTION
==================================================

GETTING THE CODE
==================================================
* To get the code:
```
git@github.com:pgniewko/Deep-Rock.git
```

* To obtain the most recent version of the code:
```
git pull origin master
```

EXTERNAL LIBRARIES
================
* CFD solver based on the lattice Boltzmann method: [Palabos](http://www.palabos.org/)

USAGE
=====

Run the Monte Carlo algorithm, and save the configurations in the file. The scripts requires the user to define the path to the `mc.py` scripts, and the path for the output files. Upon successful termination, the code produces three type of files:
* ```run.sh```        
Upon successful termination, the code produces three type of files:       
1. `.bin.txt` - file contains the binary matrix with 1 standing for the occupied site, and 0 for an empty site; file used as an image in CNN training process.          
2. `.lattice` - file contains the lattice saved in 1 line to be used in LB simulations with Palabos    
3. `.out` - file contains one line with two numbers (i) volume fraction and (ii) 1 if the packing percolates, and 0 otherwise

To run 
* ```run_lb.sh```     
This script produces `.dat` files that contain the permeability (in lattice units) and tortuosity. This data is later on used to train CNN.      


LICENSE
=======
The library is open-source. If you want to cite the library in any published work please contact me at gniewko.pablo@gmail.com for an information about credits.

COPYRIGHT NOTICE
================
Copyright (C) 2019-, Pawel Gniewek  
Email : gniewko.pablo@gmail.com  
All rights reserved.  
License: BSD 3  

REFERENCES
==========
1. []() 


ACKNOWLEDGMENTS
===============

