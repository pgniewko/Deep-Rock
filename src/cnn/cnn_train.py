#! /usr/bin/env python

import sys
import numpy as np
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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


def get_train_test_data(images, perc, values, perc_flag=True, col_=0, ts_=0.5, rs_=42):

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

def create_nn(input_shape_):
# TODO:
# 1. Add periodic padding
# 2. Make the architecture more efficient

    model = Sequential()
    model.add(Conv2D(4,  kernel_size=(3, 3), activation='relu', input_shape=input_shape_))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(),
              metrics=['mse'])

    model.summary()
    return model

if __name__ == "__main__":
# https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
# https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457
# https://www.kaggle.com/anebzt/mnist-with-cnn-in-keras-detailed-explanation
# https://jacobgil.github.io/deeplearning/filter-visualizations
# https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
# https://stats.stackexchange.com/questions/335836/cnn-architectures-for-regression
# https://datascience.stackexchange.com/questions/33725/padding-in-keras-with-output-half-sized-input
# https://datascienceplus.com/keras-regression-based-neural-networks/ 


    batch_size = 16
    epochs = 200
    # input image dimensions
    img_rows, img_cols = 64, 64
    images = get_images(sys.argv[1], img_rows, img_cols)
    porosity, perc, kappa, tau = get_values(sys.argv[2])

    input_shape = (img_rows, img_cols, 1)
    (x_train, y_train), (x_test, y_test) = get_train_test_data(images, perc, kappa, col_=2, ts_=0.33)
    

    cnn = create_nn( input_shape )

    history = cnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))  
    

    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    y_predicted = cnn.predict(x_test)
    plt.plot(y_test, y_predicted, 'o')
    plt.plot([-3,3],[-3,3],'--')
    plt.ylabel('Predicted')
    plt.xlabel('Lattice-Boltzmann')
    plt.show()

#TODO:
# 1. Save the model
