#! /usr/bin/env python
# Generally the scripts is meant to see how the network abstracts the data
# Based on the original kears codes:
# https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py 
#
# usage:
# python conv_filter_visualization.py ./model/cnn-v1.epochs-25.h5 layer_name

from __future__ import print_function

import sys
import numpy as np
import time
from keras.preprocessing.image import save_img
from keras.applications import vgg16
from keras import backend as K
from keras.models import load_model
from pbc import PeriodicPadding2D


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

if __name__ == "__main__":
    # dimensions of the generated pictures for each filter.
    img_width = 64
    img_height = 64

    model = load_model(sys.argv[1], custom_objects={'PeriodicPadding2D': PeriodicPadding2D})
    print('Model loaded.')
    model.summary()

    # the name of the layer we want to visualize
    layer_name = sys.argv[2] #'conv2d_1'

    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
    filters_number = layer_dict[layer_name].filters



    kept_filters = []
    for filter_index in range(filters_number):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        print (layer_output)
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = 1.0*np.random.randint(2, size=(1, 1, img_width, img_height))
        else:
            input_img_data = 1.0*np.random.randint(2, size=(1, img_width, img_height, 1))
    


        # we run gradient ascent for n_max=40 steps
        for i in range(40):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stich the best n^2 filters on a n x n grid.
    n = 7
    n = min(n, int(np.sqrt(filters_number)))
    n = min(n, int(np.sqrt(len(kept_filters))))

    # the filters that have the highest loss are assumed to be better-looking.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img

    # save the result to disk
    save_img('./vis/%s_stitched_filters_%dx%d.png' % (layer_name, n, n), stitched_filters)


