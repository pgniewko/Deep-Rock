#! /usr/bin/env python
#
# ./cnn_train.py ../../output/CNN/images.txt ../../output/CNN/values.txt
#

import sys
import numpy as np
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model

def get_images(fin, Lx, Ly):
    data = np.loadtxt(fin, dtype=np.dtype(int))
    n_samples, n_cols = data.shape
    images = []
    for i in range(n_samples):
        row = data[i][:]
        image = np.reshape(row, (Lx,Ly))
        image = image.astype('float32')
        images.append(image)

    return images


def get_values(fin):
    data = np.loadtxt(fin, dtype=np.dtype(float))
    data = data.T

    porosity = data[0][:]
    perc = data[1][:]
    kappa = -np.log10(data[2][:])
    tau = data[3][:]

    return porosity, perc, kappa, tau


def get_train_test_data(images, perc, values, perc_flag=True, ts_=0.5, rs_=42):

    imgs_ = []
    vals_ = []

    assert( len(images) == len(values) )
    
    for i in range(len(images)):
        if perc_flag == True:
            if perc[i] == 1:
                imgs_.append( images[i] )
                vals_.append( values[i] )
        else:
            imgs_.append( images[i] )
            vals_.append( values[i] )
            

    X_train, X_test, y_train, y_test = train_test_split(imgs_, vals_, test_size=ts_, random_state=rs_)

    x_train = np.zeros( (len(X_train), 64,64,1) )
    x_test  = np.zeros( (len(X_test), 64,64, 1) )
    for i in range(len(X_train)):
        for j in range(64):
            for k in range(64):
                x_train[i][j][k][0] = X_train[i][j][k]
    
    for i in range(len(X_test)):
        for j in range(64):
            for k in range(64):
                x_test[i][j][k][0] = X_test[i][j][k]
     
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    batch_size = 32
    epochs = 5
    # input image dimensions
    img_rows, img_cols = 64, 64
    images = get_images(sys.argv[1], img_rows, img_cols)
    porosity, perc, kappa, tau = get_values(sys.argv[2])

    input_shape = (img_rows, img_cols, 1)
    (x_train, y_train), (x_test, y_test) = get_train_test_data(images, perc, kappa, ts_=0.98, rs_=1342)

    print len(x_train), len(y_train)
    print len(x_test), len(y_test)

    filepath="./model/cnn.best.hdf5"
    cnn = load_model(filepath)
    
    y_predicted = cnn.predict(x_test)
    plt.plot(y_test, y_predicted, 'o')
    plt.plot([-3,3],[-3,3],'--')
    plt.ylabel('Predicted')
    plt.xlabel('Lattice-Boltzmann')
    plt.show()

    
    history = cnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

    
    y_predicted = cnn.predict(x_test)
    plt.plot(y_test, y_predicted, 'o')
    plt.plot([-3,3],[-3,3],'--')
    plt.ylabel('Predicted')
    plt.xlabel('Lattice-Boltzmann')
    plt.show()
