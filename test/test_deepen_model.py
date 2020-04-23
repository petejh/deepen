import unittest

from deepen import model

class DeepenModelInitTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_has_a_sensible_default_for_learning_rate(self):
        testmodel = model.Model()

        self.assertTrue(testmodel.learning_rate == 0.0075)

    def test_has_a_sensible_default_for_layer_dims(self):
        testmodel = model.Model()

        self.assertTrue(testmodel.layer_dims == [1, 1])

class DeepenModelLearningRateTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_learning_rate_is_greater_than_zero(self):
        testmodel = model.Model()

        with self.assertRaises(AssertionError):
            testmodel.learning_rate = 0

    def test_learning_rate_is_less_than_or_equal_to_one(self):
        testmodel = model.Model()

        with self.assertRaises(AssertionError):
            testmodel.learning_rate = 2

class DeepenModelLayerDimsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_all_layers_must_be_greater_than_zero(self):
        testmodel = model.Model()

        with self.assertRaises(AssertionError):
            testmodel.layer_dims = [0, 0]
