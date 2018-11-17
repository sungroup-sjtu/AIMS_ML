
import sklearn.neural_network

class MLPRegressor:

    def __init__(self, insize, outsize, layers, batch_step=None, args=dict()):

        self.layers = layers
        self.batch_step = batch_step


    def init_session(self):
        
        self.regressor = sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=self.layers,
            max_iter=(1 if not self.batch_step else self.batch_step),
            warm_start=True,
            **args
        )

    def partial_fit(self, steps, x, y_ref):

        if self.batch_step:
            print("Warning: will use batch_step (%d) as step" % self.batch_step)
            self.regressor.fit(x, y_ref)
        else:
            for j in range(steps):
                self.regressor.fit(x, y_ref)

    def predict(self, x):

        return self.regressor.predict(x)


