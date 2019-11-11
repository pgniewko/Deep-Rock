>Notice: This is research code that will not necessarily be maintained in the future.
>The code is under development so make sure you are using the most recent version.
>We welcome bug reports and PRs but make no guarantees about fixes or responses.

PROJECT DESCRIPTION
===================
(As of 11/11/19, this project is no longer maintained) This is a small research project, where I explore the application of the Convolutional Neural Network to the problem of predicting mass transport properties such as permeability or tortuosity, based on the binary 2D images. 
Technical details and partial results can be found in this tentative [draft](./paper/Deep_Rock.pdf).    


GETTING THE CODE
================
* To get the code:
```
git@github.com:pgniewko/Deep-Rock.git
```

* To obtain the most recent version of the code:
```
git pull origin master
```

EXTERNAL LIBRARIES
==================
* To run lattice Boltzmann calculations, [Palabos](http://www.palabos.org/) is required.       
* To train and run CNN, [keras](https://keras.io/) is required.      

USAGE
=====
* Run the Monte Carlo algorithm and save the configurations in the file. The scripts require that the user define the path to the `mc.py` code and the path for the output files:             
```run.sh```        
Upon successful termination, the code produces three types of files:       
    1. `.bin.txt` - file containing the binary matrix with 1 standing for the occupied site, and 0 for an empty site; the file is then used as an image in the CNN training process.          
    2. `.lattice` - file containing the lattice saved in 1 line to be used in LB simulations with Palabos    
    3. `.out` - file containing one line with two numbers: (i) volume fraction and (ii) 1 if the packing percolates, or 0 otherwise        

* Sample permeability is calculated with the lattice Boltzmann method. In order to perform this calculation, run:          
```run_lb.sh```      
    Note: To successfully run the code, the user needs to specify the path to the compiled Palabos LB simulator. This code (`./src/lb/porous-2d.cpp`) can be compiled by executing the `Makefile` (upon change of the variables in the file).         

This script produces `.dat` files that contain the permeability (in lattice units) and tortuosity. This data is later used to train CNN.      

* Prepare the input for the Conv Net.      
1. Create a list of files (`FILES_LIST.txt`) needed to run the NN training:     
```make_list.sh```     
2. Prepare two files with the set of encoded packings/images and target values:     
```./prepare_data.py FILES_LIST.txt $OUTPUT_PATH/images.txt  $OUTPUT_PATH/values.txt```    
3. Train the CNN as a regressor:      
```./cnn_train.py $OUTPUT_PATH/images.txt $OUTPUT_PATH/values.txt```
4. Once the network is trained, you can visualize the filter in a given layer with the code:     
```python conv_filter_visualization.py ./model/cnn-v1.epochs-25.h5 layer_name```     

LICENSE
=======
This project is open-source. If you want to cite the library in any published work please contact me at gniewko.pablo@gmail.com for information about credits.

COPYRIGHT NOTICE
================
Copyright (C) 2019, Pawel Gniewek  
Email : gniewko.pablo@gmail.com  
All rights reserved.  
License: BSD 3  

REFERENCES
==========
1. [Deep Rock: fluid transport properties through disordered media with convolutional neural networks](./paper/Deep_Rock.pdf) Pawel Gniewek, 2019 


ACKNOWLEDGMENTS
===============
I thank Tomek Konopczynski for help with the Keras implementation of the periodic boundary conditions padding.
