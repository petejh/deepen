from deepen import propagation as prop

class Model:

    def __init__(self, layer_dims=[1, 1], learning_rate=0.0075):
        """
        Parameters
        ----------
        layer_dims : list or tuple of int
            The number of neurons in each layer of the network.
        learning_rate : float in (0, 1]
            Learning rate for the model.
        """

        self.layer_dims = layer_dims
        self.learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert(learning_rate > 0.0 and learning_rate <= 1.0)

        self._learning_rate = learning_rate

    @property
    def layer_dims(self):
        return self._layer_dims

    @layer_dims.setter
    def layer_dims(self, layer_dims):
        assert(all(dim > 0 for dim in layer_dims))

        self._layer_dims = layer_dims

    @property
    def params(self):
        """
        params : dict of {str: ndarray}
            Final parameters for the trained model.
        """
        return self._params

    def learn_generator(self, X, Y, iterations=3000, progress=1):
        """Train the model. This generator function yields the model and cost
        after each count of iterations given by `progress`.

        Parameters
        ----------
        X : ndarray
            Input data of shape (input size, number of examples).
        Y : ndarray
            Vector of true values of shape (1, number of examples).
        iterations : int
            Maximum number of complete cycles of forward and back propagation to
            train the model.
        progress : int in [0, iterations]
            If non-zero, provide progress after each successive increment of the
            given number of iterations.

        Returns
        -------
        tuple of (int, ndarray, list)

            i : int
                Number of completed iterations.

            params : dict of {str: ndarray}
                Current parameters for the trained model.

                `Wl` : ndarray
                    Current weights matrix.
                `bl` : ndarray
                    Current biases vector.

            cost : list
                The cost computed for the current iteration of training.
        """

        # TODO: Throw error if progress < 0 or progress > iterations.

        params = prop.initialize_params(self.layer_dims)

        for i in range(1, iterations + 1):
            Y_hat, caches = prop.model_forward(X, params)

            cost = prop.compute_cost(Y_hat, Y)

            grads = prop.model_backward(Y_hat, Y, caches)
            params = prop.update_params(params, grads, self.learning_rate)
            self._params = params

            if progress and (i == 1 or i % progress == 0):
                yield (i, params, cost)

    def learn(self, X, Y, iterations=3000):
        """Train the model.

        Parameters
        ----------
        X : ndarray
            Input data of shape (input size, number of examples).
        Y : ndarray
            Vector of true values of shape (1, number of examples).
        iterations : int
            Number of complete cycles of forward and back propagation to train
            the model.

        Returns
        -------
        list of tuple of (ndarray, list)
            Parameters and cost after each iteration of training.

            params : dict of {str: ndarray}
                The parameters computed after each iteration of training.

                `Wl` : ndarray
                    Weights matrix for layer `l`.
                `bl` : ndarray
                    Biases vector for layer `l`.

            costs : list
                The cost computed after each iteration of training.
        """

        return [
            (params, cost)
            for (_, params, cost)
            in self.learn_generator(X, Y, iterations)
        ]

    def predict(self, X):
        """Calculate predictions using the trained model.

        Parameters
        ----------
        X : ndarray
            Input data of shape (input size, number of examples).

        Returns
        -------
        predictions : ndarray of {0 or 1}
            Model predictions for the given input.
        """

        predictions, _ = prop.model_forward(X, self.params)

        return predictions.round()
