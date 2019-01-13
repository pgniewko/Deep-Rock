from keras.layers import concatenate, Layer, InputSpec
def periodic_padding(image, padding=1):
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    '''
    
    upper_pad = image[:,-padding:,:]
    lower_pad = image[:,:padding,:]
    
    partial_image = concatenate([upper_pad, image, lower_pad], axis=1)
    
    left_pad = partial_image[:,:,-padding:]
    right_pad = partial_image[:,:,:padding]
    
    padded_image = concatenate([left_pad, partial_image, right_pad], axis=2)
    
    return padded_image

class PeriodicPadding2D(Layer):
    def __init__(self, padding=1, **kwargs):
        self.padding = padding
        self.input_spec = [InputSpec(ndim=4)]
        super(PeriodicPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3])

    def call(self, x, mask=None):
        pad = self.padding
        padded_image = periodic_padding(x, padding=pad)
        return padded_image
