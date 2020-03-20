import unittest
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
        self.Z_expected = np.array([[6], [13], [20]])

    def test_Z_has_the_correct_shape(self):
        Z, _ = model.linear_forward(self.A, self.W, self.b)

        self.assertEqual(Z.shape, self.Z_expected.shape)

    def test_Z_is_linear_combination_of_the_inputs(self):
        Z, _ = model.linear_forward(self.A, self.W, self.b)

        self.assertTrue(np.array_equal(Z, self.Z_expected))

    def test_cache_contains_the_inputs(self):
        _, cache = model.linear_forward(self.A, self.W, self.b)

        subtests = zip(cache, (self.A, self.W, self.b), ('A', 'W', 'b'))
        for cached, param, description in subtests:
            with self.subTest(parameter=description):
                self.assertTrue(np.array_equal(cached, param))

if __name__ == '__main__':
    unittest.main()
