import unittest
from unittest.mock import patch

import numpy as np

from deepen import model

class DeepenModelInitializeParamsTest(unittest.TestCase):
    def setUp(self):
        self.layer_dims = [2, 3, 1]
        self.num_dims = len(self.layer_dims)

    def test_returns_correct_number_of_params(self):
        number_of_params = 2 * (self.num_dims - 1)

        params = model.initialize_params(self.layer_dims)

        self.assertEqual(len(params), number_of_params)

    def test_correct_shape_for_weights(self):
        params = model.initialize_params(self.layer_dims)

        for l in range(1, self.num_dims):
            with self.subTest(l = l):
                self.assertEqual(
                    params['W' + str(l)].shape,
                    (self.layer_dims[l], self.layer_dims[l-1])
                )

    def test_correct_shape_for_biases(self):
        params = model.initialize_params(self.layer_dims)

        for l in range(1, self.num_dims):
            with self.subTest(l = l):
                self.assertEqual(
                    params['b' + str(l)].shape,
                    (self.layer_dims[l], 1)
                )

    def test_weights_are_not_zero(self):
        params = model.initialize_params(self.layer_dims)

        for l in range(1, self.num_dims):
            with self.subTest(l = l):
                self.assertTrue(params['W' + str(l)].all())

    def test_biases_are_zero(self):
        params = model.initialize_params(self.layer_dims)

        for l in range(1, self.num_dims):
            with self.subTest(l = l):
                self.assertFalse(params['b' + str(l)].any())

class DeepenModelLinearForwardTest(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1], [2]])
        self.W = np.array([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([[1], [2], [3]])

        self.params = (self.A, self.W, self.b)

        # Z = W·A + b
        self.Z_expected = np.array([[6], [13], [20]])

    def test_Z_has_the_correct_shape(self):
        Z, _ = model.linear_forward(*self.params)

        self.assertEqual(Z.shape, self.Z_expected.shape)

    def test_Z_is_linear_combination_of_the_inputs(self):
        Z, _ = model.linear_forward(*self.params)

        self.assertTrue(np.array_equal(Z, self.Z_expected))

    def test_cache_contains_the_inputs(self):
        _, cache = model.linear_forward(*self.params)

        subtests = zip(cache, self.params, ('A', 'W', 'b'))
        for cached, param, description in subtests:
            with self.subTest(parameter=description):
                self.assertTrue(np.array_equal(cached, param))

class DeepenModelLayerForwardTest(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1], [2]])
        self.W = np.array([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([[1], [2], [3]])

        self.params = (self.A, self.W, self.b)

        # Z = W·A + b
        self.Z = np.array([[6], [13], [20]])
        # relu(Z)
        self.relu = np.array([[6], [13], [20]])
        # sigmoid(Z)
        self.sigmoid = np.array([[0.99752738], [0.99999774], [1.0]])

    def test_A_has_the_correct_shape(self):
        A, _ = model.layer_forward(*self.params, 'relu')

        self.assertTrue(A.shape == self.Z.shape)

    def test_linear_cache_contains_the_inputs(self):
        _, (linear_cache, _) = model.layer_forward(*self.params, 'relu')

        subtests = zip(linear_cache, self.params, ('A', 'W', 'b'))
        for cached, param, description in subtests:
            with self.subTest(parameter=description):
                self.assertTrue(np.array_equal(cached, param))

    def test_activation_cache_has_the_correct_shape(self):
        _, (_, activation_cache) = model.layer_forward(*self.params, 'relu')

        self.assertTrue(activation_cache.shape == self.Z.shape)

    def test_calls_relu_activation(self):
        relu_returns = (self.relu, self.Z)

        with unittest.mock.patch(
            'deepen.model.relu',
            return_value = relu_returns
        ) as relu:
            model.layer_forward(*self.params, 'relu')

            relu.assert_called_once()

    def test_calls_sigmoid(self):
        sigmoid_returns = (self.sigmoid, self.Z)

        with unittest.mock.patch(
            'deepen.model.sigmoid',
            return_value = sigmoid_returns
        ) as sigmoid:
            model.layer_forward(*self.params, 'sigmoid')

            sigmoid.assert_called_once()

class DeepenModelModelForwardTest(unittest.TestCase):
    def setUp(self):
        self.X = np.ones((2, 1))

        self.W1 = np.ones((3, 2))
        self.W2 = np.ones((3, 3))
        self.W3 = np.ones((1, 3))
        self.b1 = np.zeros((3, 1))
        self.b2 = np.zeros((3, 1))
        self.b3 = np.zeros((1, 1))
        self.params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3
        }

    def test_cache_contains_L_caches(self):
        _, caches = model.model_forward(self.X, self.params)

        self.assertTrue(len(caches) == len(self.params) // 2)

    def test_Y_hat_has_the_correct_shape(self):
        Y_hat, _ = model.model_forward(self.X, self.params)

        self.assertTrue(Y_hat.shape == (1, self.X.shape[1]))

    def test_calls_relu_activation_L_minus_1_times(self):
        with unittest.mock.patch(
            'deepen.model.relu',
            wraps = model.relu
        ) as relu_spy:
            model.model_forward(self.X, self.params)

            self.assertTrue(relu_spy.call_count == len(self.params) // 2 - 1)

    def test_calls_sigmoid_activation_one_time(self):
        with unittest.mock.patch(
            'deepen.model.sigmoid',
            wraps = model.sigmoid
        ) as sigmoid_spy:
            model.model_forward(self.X, self.params)

            sigmoid_spy.assert_called_once()

class DeepenModelComputeCostTest(unittest.TestCase):
    def setUp(self):
        self.Y_hat = np.array([[0.8, 0.1, 0.9]])
        self.Y = np.array([[1, 0, 1]])
        self.expected_cost = 0.14462152

    def test_cost_has_the_correct_shape(self):
        cost = model.compute_cost(self.Y_hat, self.Y)

        self.assertTrue(cost.shape == ())

    def test_computes_the_cost(self):
        cost = model.compute_cost(self.Y_hat, self.Y)

        self.assertAlmostEqual(cost, self.expected_cost)

if __name__ == '__main__':
    unittest.main()
