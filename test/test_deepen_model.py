import unittest

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

if __name__ == '__main__':
    unittest.main()
