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
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sb

from pbc import PeriodicPadding2D

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
    x_test  = np.zeros( (len(X_test), 64,64,1) )
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
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape_))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['mse'])

    model.summary()
    return model, "cnn"

def create_nn_pbc(input_shape_):
    model = Sequential()

    #Layer No. 0
    model.add(PeriodicPadding2D(padding=1, input_shape=input_shape_,))
    
    #Layer No. 1
    model.add(Conv2D(64, 
                     kernel_size=(3, 3), 
#                     input_shape=input_shape_,
                     padding='valid',
                     activation='relu'))
    model.add(PeriodicPadding2D(padding=1))

    # Layer No. 2
    model.add(Conv2D(128, 
                     kernel_size=(3, 3), 
                     padding='valid',
                     activation='relu'))
    model.add(PeriodicPadding2D(padding=1))
   
    # Layer No. 3 
    model.add(Conv2D(256, 
                     kernel_size=(3, 3), 
                     padding='valid',
                     activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(PeriodicPadding2D(padding=1))
    model.add(Dropout(0.25))


    # Layer No. 4
    model.add(Conv2D(64, 
                     kernel_size=(3, 3), 
                     padding='valid',
                     activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(PeriodicPadding2D(padding=1)) 
    model.add(Dropout(0.25))

    # Layer No. 5
    model.add(Conv2D(64, 
                     kernel_size=(3, 3), 
                     padding='valid',
                     activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
   
    # Layer No. 6
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    
    # Layer No. 7
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.05))

    # Layer No. 8
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['mse'])

    model.summary()
    
    return model, "cnn-pbc"



def save_history(train_0, test_0, history, fname="history_loss.log"):
    v_train_0 = train_0[0]
    v_test_0  = test_0[0]
    vals_train = history.history['loss']
    vals_test  = history.history['val_loss'] 

    vals_train.insert(0, v_train_0)
    vals_test.insert(0, v_test_0)

    fout = open(fname, 'w')
    for i in range( len(vals_train) ):
        fout.write( str(i) + " " + str(vals_train[i]) + " " + str(vals_test[i]) + "\n" ) 

    fout.close()


def save_preds(y_test, y_pred, fname='predictions.log'):
    fout = open(fname, 'w')
    for i, y_el in enumerate(y_test):
        fout.write( str(y_el) + " " + str(y_pred[i][0]) + "\n" )

    fout.close()


if __name__ == "__main__":
# https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457
# https://www.kaggle.com/anebzt/mnist-with-cnn-in-keras-detailed-explanation
# https://jacobgil.github.io/deeplearning/filter-visualizations
# https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
# https://stats.stackexchange.com/questions/335836/cnn-architectures-for-regression
# https://datascience.stackexchange.com/questions/33725/padding-in-keras-with-output-half-sized-input
# https://datascienceplus.com/keras-regression-based-neural-networks/ 
# https://machinelearningmastery.com/check-point-deep-learning-models-keras/


    batch_size = 32
    epochs = 150 
    img_rows, img_cols = 64, 64
    images = get_images(sys.argv[1], img_rows, img_cols)
    porosity, perc, kappa, tau = get_values(sys.argv[2])

    input_shape = (img_rows, img_cols, 1)
    (x_train, y_train), (x_test, y_test) = get_train_test_data(images, perc, kappa, ts_=0.5, rs_=1342)


#    cnn, model_name = create_nn(input_shape)
    cnn, model_name = create_nn_pbc(input_shape)

    filepath="./model/%s.best.hdf5" %(model_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history_train_epoch_0 = cnn.evaluate(x_train, y_train, batch_size=batch_size)
    history_test_epoch_0 = cnn.evaluate(x_test, y_test, batch_size=batch_size)
 
    history = cnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(x_test, y_test))  
    
    cnn.save("./model/%s.epochs-%d.hdf5" % (model_name, epochs) )
    y_pred = cnn.predict(x_test)
    
    # Save data #
    f_history = "./output_data/%s-history_loss.log" %(model_name)
    save_history(history_train_epoch_0, history_test_epoch_0, history, f_history)
    
    f_preds = "./output_data/%s-predictions.log" %(model_name)
    save_preds(y_test, y_pred, f_preds)
 
    
    # Plot data - if flat True
    plot_flag = False
    if plot_flag:
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        # Permeability
        plt.plot(y_test, y_pred, 'o')
        plt.plot([-3,3],[-3,3],'--')
        plt.ylabel('Predicted')
        plt.xlabel('Lattice-Boltzmann')
        plt.show()


