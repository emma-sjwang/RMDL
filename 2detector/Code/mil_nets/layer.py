from keras.layers import Layer
from keras import backend as K
from . import pooling_method as pooling


class ADD(Layer):
    def __init__(self, pooling_mode='add', **kwargs):
        self.pooling_mode = pooling_mode

        super(ADD, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_built = True

    def call(self, x, mask=None):
        # do-pooling operator
        x =pooling.choice_pooling(x, self.pooling_mode)
        return x

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[1] = 1
        return tuple(shape)

    def get_config(self):
        config = {
            'pooling_mode': self.pooling_mode
        }
        base_config = super(ADD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Expand_dims(Layer):
    def __init__(self, dims = 200, **kwargs):
        self.dims = dims
        super(Expand_dims, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_built = True

    def call(self, x):
        x = K.squeeze(x, 1)
        return  K.repeat(x, self.dims)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[1] = self.dims
        return tuple(shape)

    def get_config(self):
        config = {
            'dims': self.dims
        }
        base_config = super(Expand_dims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

