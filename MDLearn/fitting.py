
import numpy as np 

backend = 'tch'


class PerceptronFitter:
    """ A simple wrapper for regressor.MLPRegressor
    """

    def __init__(self, insize, outsize, layers, batch_size=None, batch_step=None, args_layer=dict(), args_opt=dict()):
        """ A simple wrapper for regressor.MLPRegressor
            Additional args will be passed to sklearn.neural.MLPRegreesor()
        """

        if backend == 'tf':
            from . import tf
            self.regressor = tf.regression.MLPRegressor(insize, outsize, layers, batch_size, batch_step, args_layer, args_opt)
            #self.regressor = tf.regression.DNNRegressor(insize, outsize, layers, batch_size, batch_step, args_layer, args_opt)
        elif backend == 'skl':
            from . import skl
            args = dict.copy(args_layer)
            args.update(args_opt)
            self.regressor = skl.regression.MLPRegressor(insize, outsize, layers, batch_step, args)

        elif backend == 'tch':
            from . import tch
            self.regressor = tch.regression.MLPRegressor(insize, outsize, layers, batch_size, batch_step, args_layer, args_opt)

        else:
            raise NotImplementedError()
            
        self._has_init = False 

    def train(self, steps, data_x, data_y):

        if not self._has_init:
            self.regressor.init_session()
            self._has_init = True 

        self.regressor.partial_fit(steps, data_x, data_y)

    def predict_batch(self, batch_x):
        """ Predict a lot of data at the same time.
        """
        return self.regressor.predict(batch_x)
