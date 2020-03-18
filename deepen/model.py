import numpy as np
from deepen.activation import relu, sigmoid

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

def linear_forward(A, W, b):
    """Calculate the linear part of forward propagation for the current layer.

    .. math:: $$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}, where $A^{[0]} = X$

    Parameters
    ----------
    A : ndarray
        Activations from the previous layer, of shape (size of previous layer,
        number of examples).
    W : ndarray
        Weights matrix of shape (size of current layer, size of previous layer).
    b : ndarray
        Bias vector of shape (size of current layer, 1).

    Returns
    -------
    Z : ndarray
        Input of the activation function, also called pre-activation parameter,
        of shape (size of current layer, number of examples).
    cache : tuple of ndarray
        Store `A`, `W`, and `b` for computing the backward pass efficiently.
    """

    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache

def layer_forward(A_prev, W, b, activation):
    """Compute forward propagation for a single layer.

    Parameters
    ----------
    A_prev : ndarray
        Activations from the previous layer of shape (size of previous layer,
        number of examples).
    W : ndarray
        Weights matrix of shape (size of current layer, size of previous layer).
    b : ndarray
        Bias vector of shape (size of the current layer, 1).
    activation : str {"sigmoid", "relu"}
        Activation function to be used in this layer.

    Returns
    -------
    A : ndarray
        Output of the activation function of shape (size of current layer,
        number of examples).
    cache : tuple of (tuple of ndarray, ndarray)
        Stored for computing the backward pass efficiently.

        linear_cache : tuple of ndarray
            Stores `cache` returned by `linear_forward()`.
        activation_cache : ndarray
            Stores `Z` returned by 'linear_forward()`.
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache
