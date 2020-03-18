import numpy as np

def sigmoid(Z):
    """Compute the sigmoid activation function.

    Parameters
    ----------
    Z : ndarray
        Pre-activation parameters for the current layer.

    Returns
    -------
    A : ndarry
        Post-activation parameters for the current layer. Same shape as `Z`.
    ndarray
        Store `Z` for computing the backward pass efficiently.
    """

    A = 1 / (1 + np.exp(-Z))

    return A, Z

def relu(Z):
    """Compute the Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    Z : ndarray
        Pre-activation parameters for the current layer.

    Returns
    -------
    A : ndarry
        Post-activation parameters for the current layer. Same shape as `Z`.
    ndarray
        Store `Z` for computing the backward pass efficiently.
    """

    A = np.maximum(0, Z)

    return A, Z