
import tensorflow as tf 
import numpy as np 


class MLPRegressor:

    def __init__(self, insize, outsize, layers, batch_size=None, batch_step=None, args_layer=dict(), args_opt=dict()):
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.batch_step = batch_step
        with self.graph.as_default():
            self._init_layer(insize, outsize, layers, **args_layer)
            self._init_optimizer(**args_opt)
            self.saver = tf.train.Saver()


    def _init_layer(self, insize, outsize, layers, **kwargs):
        """ Initialize all layers. (Including input & output)
        Args:
        ---
        insize: Size of input.
        outsize: Size of output.
        layers: Tuple (axb), size of each hidden layer.
        kwargs will be passed into _add_layer.
        """

        with self.graph.as_default():

            def _add_layer(inputs, outsize, activator=None, init_stddev=2.0, init_intercept=0.2):
                """ Add one linear layer.
                Args:
                ---
                inputs: tf.Tensor object. Must be 2d matrix.
                outsize: Size of neuron number.
                activator: Type of activator. Set `None` to use linear combination only.
                """
                a = int(inputs.shape[1])
                W = tf.Variable(tf.random_normal([a, outsize], stddev=init_stddev))   # weight
                b = tf.Variable(tf.random_normal([outsize], stddev=init_stddev) + init_intercept)

                if activator is None:
                    return tf.add(tf.matmul(inputs, W), b)     # simply transformation
                else:
                    return activator(tf.add(tf.matmul(inputs, W), b))

            vartype = kwargs.pop('type', tf.float32)

            if self.batch_size is None:

                self.x = tf.placeholder(vartype, [None, insize])
                self.y_ref = tf.placeholder(vartype, [None, outsize])

            else:
                pass

            nnlayers = []

            if len(layers) == 0:
                nnlayers.append(_add_layer(self.x, outsize, **kwargs))

            elif len(layers) >= 1:
                nnlayers.append(_add_layer(self.x, layers[0], **kwargs))
                for i in range(1, len(layers)):
                    nnlayers.append(_add_layer(nnlayers[i-1], layers[i], **kwargs))
                # last layer is dense
                if 'activator' in kwargs:
                    kwargs['activator'] = None
                nnlayers.append(_add_layer(nnlayers[-1], outsize, **kwargs))

            self.y = nnlayers[-1]


    def _init_optimizer(self, optimizer=tf.train.AdamOptimizer, learning_rate=0.01, **kwargs):

        with self.graph.as_default():

            self.loss = tf.losses.mean_squared_error(self.y_ref, self.y)
            self.optimizer = optimizer(learning_rate, **kwargs)
            self.train = self.optimizer.minimize(self.loss)

    def init_session(self):

        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.session.run(init)

    def partial_fit(self, steps, x, y_ref):

        for j in range(steps):

            self.session.run(self.train, feed_dict={self.x:x, self.y_ref:y_ref})

    def predict(self, x):

        return self.session.run(self.y, feed_dict={self.x:x})

    def loss(self, x, y_ref):

        return self.session.run(self.loss, feed_dict={self.x:x, self.y_ref:y_ref})

    def save(self, filename):

        with self.session as sess:
            path = self.saver.save(sess, filename)
            print('Model saved in path: %s' % path)

    def load(self, filename):

        with self.session as sess:
            self.saver.restore(sess, filename)
            print('Load model from %s' % filename)


