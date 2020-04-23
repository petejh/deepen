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

    def learn(self, X, Y, iterations=3000):
        """Train the model.

        Parameters
        ----------
        X : ndarray
            Input data of shape (input size, number of examples).
        Y : ndarray
            Vector of true values of shape (1, number of examples).
        iterations : int
            Number of complete cycles of forward and back propagation to train the
            model.

        Returns
        -------
        tuple of (ndarray, list)

            params : dict of {str: ndarray}
                Final parameters for the trained model.

                `Wl` : ndarray
                    Final weights matrix.
                `bl` : ndarray
                    Final biases vector.

            costs : list
                The cost computed for each iteration of training.
        """

        costs = []
        params = prop.initialize_params(self.layer_dims)

        for i in range(0, iterations):
            Y_hat, caches = prop.model_forward(X, params)

            cost = prop.compute_cost(Y_hat, Y)
            costs.append(cost)

            grads = prop.model_backward(Y_hat, Y, caches)
            params = prop.update_params(params, grads, self.learning_rate)

        self._params = params
        return (params, costs)

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
