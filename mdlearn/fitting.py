
class PerceptronFitter:
    """ A simple wrapper for regressor.MLPRegressor
    """

    def __init__(self, insize, outsize, layers, batch_size=None, batch_step=None, args_layer=dict(), args_opt=dict()):
        """ A simple wrapper for regressor.MLPRegressor
            Additional args will be passed to sklearn.neural.MLPRegreesor()
        """

        from . import tch
        self.regressor = tch.regression.MLPRegressor(insize, outsize, layers, batch_size, batch_step, args_layer, args_opt)

        self._has_init = False

    def predict_batch(self, batch_x):
        """ Predict a lot of data at the same time.
        """
        return self.regressor.predict(batch_x)
