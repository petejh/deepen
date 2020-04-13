import numpy as np
from deepen.activation import relu, relu_backward, sigmoid, sigmoid_backward

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
        params['W' + str(l)] = (
            np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        )
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

def model_forward(X, parameters):
    """Compute forward propagation for [LINEAR->RELU]*(L-1) -> [LINEAR->SIGMOID].

    Parameters
    ----------
    X : ndarray
        Input data of shape (input size, number of examples)
    parameters : dict of {str: ndarray}
        Output of initialize_parameters_deep()

    Returns
    -------
    Y_hat : ndarray
        Vector of prediction probabilities of shape (1, number of
        examples).
    caches : list of (tuple of (tuple of ndarray, ndarray))
        The L `cache` results from `layer_forward()`.
    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = layer_forward(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            "relu"
        )
        caches.append(cache)

    Y_hat, cache = layer_forward(
        A,
        parameters["W" + str(L)],
        parameters["b" + str(L)],
        "sigmoid"
    )
    caches.append(cache)

    return Y_hat, caches

def compute_cost(Y_hat, Y):
    """Compute the cross-entropy cost.

    .. math:: $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right))

    Parameters
    ----------
    Y_hat : ndarray
        Vector of prediction probabilities from `model_forward()` of shape
        (1, number of examples).
    Y : ndarray
        Vector of true values of shape (1, number of examples).

    Returns
    -------
    cost : list of int
        Cross-entropy cost.
    """

    m = Y.shape[1]

    cost = (1./m) * (-np.dot(Y, np.log(Y_hat).T) - np.dot(1-Y, np.log(1-Y_hat).T))
    cost = np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):
    """Calculate the linear portion of backward propagation for a single layer.

    Parameters
    ----------
    dZ : ndarray
        Gradient of the cost with respect to the linear output of layer l.
    cache : tuple of ndarray
        Stored `A`, `W`, `b` from `linear_forward()`.

    Returns
    -------
    dA_prev : ndarray
        Gradient of the cost with respect to the activation of the previous
        layer, l-1. Shape of `cache['A']`.
    dW : ndarray
        Gradient of the cost with respect to W for the current layer, l. Shape
        of `cache['W']`.
    db : ndarray
        Gradient of the cost with respect to b for the current layer, l. Shape
        of `cache['b']`.
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def layer_backward(dA, cache, activation):
    """Compute backward propagation for a single layer.

    Parameters
    ----------
    dA: ndarray
        Post-activation gradient for current layer, l.
    cache : tuple of (tuple of ndarray, ndarray)
        Stored `(linear_cache, activation_cache)` from `layer_forward()`.
    activation : str {"relu", "sigmoid"}
        Activation function to be used in this layer.

    Returns
    -------
    dA_prev : ndarray
        Gradient of the cost with respect to the activation of the previous
        layer, l-1. Shape of `cache['A']`.
    dW : ndarray
        Gradient of the cost with respect to W for the current layer, l. Shape
        of `cache['W']`.
    db : ndarray
        Gradient of the cost with respect to b for the current layer, l. Shape
        of `cache['b']`.
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def model_backward(Y_hat, Y, caches):
    """Compute backward propagation for [LINEAR->RELU]*(L-1) -> [LINEAR->SIGMOID].

    Parameters
    ----------
    Y_hat : ndarray
        Vector of prediction probabilities from `model_forward()` of shape
        (1, number of examples).
    Y : ndarray
        Vector of true values of shape (1, number of examples).
    caches : list of (tuple of (tuple of ndarray, ndarray))
        Stored results of `model_forward()`.

    Returns
    -------
    grads : dict of {str: ndarray}
        Gradients for layer `l` in `range(L-1)`.

        dAl : ndarray
            Gradient of the activations for layer `l`.
        dWl : ndarray
            Gradient of the weights for layer `l`.
        dbl : ndarray
            Gradient of the biases for layer `l`.
    """

    grads = {}
    L = len(caches)
    m = Y_hat.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dY_hat = -(np.divide(Y, Y_hat) - np.divide(1-Y, 1-Y_hat))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = (
        layer_backward(dY_hat, current_cache, "sigmoid")
    )

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = (
            layer_backward(grads["dA" + str(l+1)], current_cache, "relu")
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_params(params, grads, learning_rate):
    """Update parameters using gradient descent.

    Parameters
    ----------
    params : dict of {str: ndarray}
        Initialized parameters from `intialize_params()`.
    grads : dict of {str: ndarray}
        Gradients from `model_backward()`.
    learning_rate : float in (0, 1)
        Learning rate for the model.

    Returns
    -------
    params : dict of {str: ndarray}
        Updated parameters.

        `Wl` : ndarray
            Updated weights matrix.
        `bl` : ndarray
            Updated biases vector.
    """

    L = len(params) // 2

    for l in range(L):
        params["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        params["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return params

def learn(X, Y, layer_dims, learning_rate=0.0075, iterations=3000):
    """Run the model.

    Parameters
    ----------
    X : ndarray
        Input data of shape (input size, number of examples).
    Y : ndarray
        Vector of true values of shape (1, number of examples).
    layer_dims : list or tuple of int
        The number of neurons in each layer of the network.
    learning_rate : float in (0, 1]
        Learning rate for the model.
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
    params = initialize_params(layer_dims)

    for i in range(0, iterations):
        Y_hat, caches = model_forward(X, params)

        cost = compute_cost(Y_hat, Y)
        costs.append(cost)

        grads = model_backward(Y_hat, Y, caches)
        params = update_params(params, grads, learning_rate)

    return (params, costs)
