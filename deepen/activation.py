import numpy as np

def relu(Z):
    """Compute the Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    Z : ndarray
        Pre-activation parameters for the current layer.

    Returns
    -------
    A : ndarray
        Post-activation parameters for the current layer. Same shape as `Z`.
    ndarray
        Store `Z` for computing the backward pass efficiently.
    """

    A = np.maximum(0, Z)

    return A, Z

def relu_backward(dA, cache):
    """Compute backward propagation through the RELU activation function.

    Parameters
    ----------
    dA : ndarray
        Post-activation gradients for the current layer.
    cache : ndarray
        Stored `Z` from `relu()`.

    Returns
    -------
    dZ : ndarray
        Gradients of the cost with respect to `Z`.
    """

    Z = cache

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ

def sigmoid(Z):
    """Compute the sigmoid activation function.

    Parameters
    ----------
    Z : ndarray
        Pre-activation parameters for the current layer.

    Returns
    -------
    A : ndarray
        Post-activation parameters for the current layer. Same shape as `Z`.
    ndarray
        Store `Z` for computing the backward pass efficiently.
    """

    A = 1 / (1 + np.exp(-Z))

    return A, Z

def sigmoid_backward(dA, cache):
    """Compute backward propagation through the sigmoid activation function.

    Parameters
    ----------
    dA : ndarray
        Post-activation gradients for the current layer.
    cache : ndarray
        Stored `Z` fron `sigmoid()`.

    Returns
    -------
    dZ : ndarray
        Gradients of the cost with respect to `Z`.
    """

    Z = cache

    A, _ = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ
