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
            with self.subTest(parameter = description):
                self.assertTrue(np.array_equal(cached, param))

class DeepenModelLayerForwardTest(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1], [2]])
        self.W = np.array([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([[1], [2], [3]])

        self.params = (self.A, self.W, self.b)

        # Z = W·A + b
        self.Z = np.array([[6], [13], [20]])

    def test_A_has_the_correct_shape(self):
        A, _ = model.layer_forward(*self.params, 'relu')

        self.assertTrue(A.shape == self.Z.shape)

    def test_linear_cache_contains_the_inputs(self):
        _, (linear_cache, _) = model.layer_forward(*self.params, 'relu')

        subtests = zip(linear_cache, self.params, ('A', 'W', 'b'))
        for cached, param, description in subtests:
            with self.subTest(parameter = description):
                self.assertTrue(np.array_equal(cached, param))

    def test_activation_cache_has_the_correct_shape(self):
        _, (_, activation_cache) = model.layer_forward(*self.params, 'relu')

        self.assertTrue(activation_cache.shape == self.Z.shape)

    def test_calls_relu_activation(self):
        with unittest.mock.patch(
            'deepen.model.relu',
            wraps = model.relu
        ) as relu_spy:
            model.layer_forward(*self.params, 'relu')

            relu_spy.assert_called_once()

    def test_calls_sigmoid(self):
        with unittest.mock.patch(
            'deepen.model.sigmoid',
            wraps = model.sigmoid
        ) as sigmoid_spy:
            model.layer_forward(*self.params, 'sigmoid')

            sigmoid_spy.assert_called_once()

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

class DeepenModelLinearBackward(unittest.TestCase):
    def setUp(self):
        self.A_prev = np.array([[1], [2]])
        self.W = np.array([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([[1], [2], [3]])
        self.dZ = np.array([[1], [2], [3]])

        self.params = (self.dZ, (self.A_prev, self.W, self.b))

        self.dA_expected = np.array([[22], [28]])
        self.dW_expected = np.array([[1, 2], [2, 4], [3, 6]])
        self.db_expected = np.array([[1], [2], [3]])

        self.expected = (self.dA_expected, self.dW_expected, self.db_expected)

    def test_gradients_have_the_correct_shape(self):
        dA, dW, db = model.linear_backward(*self.params)

        subtests = zip((dA, dW, db), self.expected, ('dA', 'dW', 'db'))
        for grad, expected, description in subtests:
            with self.subTest(gradient = description):
                self.assertTrue(grad.shape == expected.shape)

    def test_computes_the_gradients(self):
        dA, dW, db = model.linear_backward(*self.params)

        subtests = zip((dA, dW, db), self.expected, ('dA', 'dW', 'db'))
        for grad, expected, description in subtests:
            with self.subTest(gradient = description):
                self.assertTrue(np.array_equal(grad, expected))

class DeepenModelLayerBackwardTest(unittest.TestCase):
    def setUp(self):
        self.dA = np.ones((3,1))

        self.A= np.array([[1], [2]])
        self.W = np.array([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([[1], [2], [3]])
        self.Z = np.array([[6], [13], [20]])
        self.cache = ((self.A, self.W, self.b), self.Z)

        self.dA_expected = np.array([[9.], [12.]])
        self.dW_expected = np.array([[1., 2.], [1., 2.], [1., 2.]])
        self.db_expected = np.array([[1.], [1.], [1.]])

        self.expected = (self.dA_expected, self.dW_expected, self.db_expected)

    def test_gradients_have_the_correct_shape(self):
        dA, dW, db = model.layer_backward(self.dA, self.cache, 'relu')

        subtests = zip((dA, dW, db), self.expected, ('dA', 'dW', 'db'))
        for grad, expected, description in subtests:
            with self.subTest(gradient = description):
                self.assertTrue(grad.shape == expected.shape)

    def test_computes_the_gradients(self):
        dA, dW, db = model.layer_backward(self.dA, self.cache, 'relu')

        subtests = zip((dA, dW, db), self.expected, ('dA', 'dW', 'db'))
        for grad, expected, description in subtests:
            with self.subTest(gradient = description):
                self.assertTrue(np.array_equal(grad, expected))

    def test_calls_relu_backward(self):
        with unittest.mock.patch(
            'deepen.model.relu_backward',
            wraps = model.relu_backward
        ) as relu_spy:
            model.layer_backward(self.dA, self.cache, 'relu')

            relu_spy.assert_called_once()

    def test_calls_sigmoid_backward(self):
        with unittest.mock.patch(
            'deepen.model.sigmoid_backward',
            wraps = model.sigmoid_backward
        ) as sigmoid_spy:
            model.layer_backward(self.dA, self.cache, 'sigmoid')

            sigmoid_spy.assert_called_once()

class DeepenModelModelBackwardTest(unittest.TestCase):
    def setUp(self):
        self.Y_hat = np.array([[0.99999998]])
        self.Y = np.array([[1]])

        self.X = np.ones((2, 1))
        self.W1 = np.ones((3, 2))
        self.b1 = np.zeros((3, 1))
        self.A1 = np.array([[2.], [2.], [2.]])
        self.W2 = np.ones((3, 3))
        self.b2 = np.zeros((3, 1))
        self.A2 = np.array([[6.], [6.], [6.]])
        self.W3 = np.ones((1, 3))
        self.b3 = np.zeros((1, 1))
        self.A3 = np.array([[18.]])
        self.caches = [
            ((self.X, self.W1, self.b1), self.A1),
            ((self.A1, self.W2, self.b2), self.A2),
            ((self.A2, self.W3, self.b3), self.A3)
        ]

        self.grads_expected = {
            "dA2": np.array([[-1.52299795e-08], [-1.52299795e-08], [-1.52299795e-08]]),
            "dW3": np.array([[-9.1379877e-08, -9.1379877e-08, -9.1379877e-08]]),
            "db3": np.array([[-1.52299795e-08]]),
            "dA1": np.array([[-4.56899385e-08], [-4.56899385e-08], [-4.56899385e-08]]),
            "dW2": np.array(
                [[-3.0459959e-08, -3.0459959e-08, -3.0459959e-08],
                [-3.0459959e-08, -3.0459959e-08, -3.0459959e-08],
                [-3.0459959e-08, -3.0459959e-08, -3.0459959e-08]]
            ),
            "db2": np.array([[-1.52299795e-08], [-1.52299795e-08], [-1.52299795e-08]]),
            "dA0": np.array([[-1.37069815e-07], [-1.37069815e-07]]),
            "dW1": np.array(
                [[-4.56899385e-08, -4.56899385e-08],
                [-4.56899385e-08, -4.56899385e-08],
                [-4.56899385e-08, -4.56899385e-08]]
            ),
            "db1": np.array([[-4.56899385e-08], [-4.56899385e-08], [-4.56899385e-08]])
        }

    def test_grads_has_the_correct_length(self):
        grads = model.model_backward(self.Y_hat, self.Y, self.caches)

        self.assertTrue(len(grads) == len(self.grads_expected))

    def test_grads_have_the_correct_shape(self):
        grads = model.model_backward(self.Y_hat, self.Y, self.caches)

        test_labels = ('dA2', 'dW3', 'db3', 'dA1', 'dW2', 'db2', 'dA0', 'dW1', 'db1')
        for gradient in test_labels:
            with self.subTest(gradient = gradient):
                self.assertTrue(grads[gradient].shape == self.grads_expected[gradient].shape)

    def test_computes_the_gradients(self):
        grads = model.model_backward(self.Y_hat, self.Y, self.caches)

        test_labels = ('dA2', 'dW3', 'db3', 'dA1', 'dW2', 'db2', 'dA0', 'dW1', 'db1')
        for gradient in test_labels:
            with self.subTest(gradient = gradient):
                self.assertTrue(np.allclose(grads[gradient], self.grads_expected[gradient]))

    def test_calls_relu_activation_L_minus_1_times(self):
        with unittest.mock.patch(
            'deepen.model.relu_backward',
            wraps = model.relu_backward
        ) as relu_spy:
            model.model_backward(self.Y_hat, self.Y, self.caches)

            self.assertTrue(relu_spy.call_count == len(self.caches)  - 1)

    def test_calls_sigmoid_activation_one_time(self):
        with unittest.mock.patch(
            'deepen.model.sigmoid_backward',
            wraps = model.sigmoid_backward
        ) as sigmoid_spy:
            model.model_backward(self.Y_hat, self.Y, self.caches)

            sigmoid_spy.assert_called_once()

if __name__ == '__main__':
    unittest.main()
