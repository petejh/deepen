import numpy as np

def initialize_params(layer_dims):
    """Create and initialize the params of an L-layer neural network.

    Parameters
    ----------
    layer_dims : list or tuple of int
        The number of neurons in each layer of the network.

    Returns
    -------
    params : dict of {str: ndarray}
        Initialized parameters for each layer, l, of the L-layer network.

        Wl : ndarray
            Weights matrix of shape (`layer_dims[l]`, `layer_dims[l-1]`).
        bl : ndarray
            Biases vector of shape (`layer_dims[l]`, 1).
    """

    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params
