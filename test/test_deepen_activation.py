import unittest

import numpy as np

from deepen import activation

class DeepenActivationReluTest(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[1], [2], [3]])
        self.reluZ = np.array([[1], [2], [3]])

    def test_returns_the_transformed_input(self):
        Z, _ = activation.relu(self.Z)

        self.assertTrue(np.array_equal(Z, self.reluZ))

    def test_cache_contains_the_input(self):
        _, cache = activation.relu(self.Z)

        self.assertTrue(np.array_equal(cache, self.Z))

class DeepenActivationSigmoidTest(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[1], [2], [3]])
        self.sigmoidZ = np.array([[0.73105858], [0.88079708], [0.95257413]])

    def test_returns_the_transformed_input(self):
        Z, _ = activation.sigmoid(self.Z)

        self.assertTrue(np.allclose(Z, self.sigmoidZ))

    def test_cache_contains_the_input(self):
        _, cache = activation.sigmoid(self.Z)

        self.assertTrue(np.array_equal(cache, self.Z))

if __name__ == '__main__':
    unittest.main()
