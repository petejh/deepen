import unittest

import numpy as np

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

class DeepenModelLearnGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.testmodel = model.Model()
        self.X = np.ones((1,1))
        self.Y = np.ones((1,1))
        self.iterations = 100
        self.interval = 10

    def test_completes_one_iteration(self):
        generator = self.testmodel.learn_generator(self.X, self.Y, self.iterations, self.interval)

        (iteration, _, _) = next(generator)

        self.assertTrue(iteration == 1)

    def test_yields_cost_for_the_first_iteration(self):
        generator = self.testmodel.learn_generator(self.X, self.Y, self.iterations, self.interval)

        (_, _, cost) = next(generator)

        self.assertIsNotNone(cost)

    def test_yields_intermediate_cost_for_each_interval(self):
        intermediate_costs = []

        generator = self.testmodel.learn_generator(self.X, self.Y, self.iterations, self.interval)

        for (_, _, cost) in generator:
            intermediate_costs.append(cost)

        self.assertTrue(len(intermediate_costs) == int(self.iterations / self.interval + 1))

    def test_yields_intermediate_results_for_last_iteration_with_whole_intervals(self):
        intermediates = []

        generator = self.testmodel.learn_generator(self.X, self.Y, self.iterations, self.interval)

        for (iteration, _, _) in generator:
            intermediates.append(iteration)

        self.assertTrue(intermediates[-1] == self.iterations)

    def test_does_not_return_intermediate_results_when_interval_is_0(self):
        intermediate_costs = []
        interval = 0

        generator = self.testmodel.learn_generator(self.X, self.Y, self.iterations, interval)

        for (_, _, cost) in generator:
            intermediate_costs.append(cost)

        self.assertTrue(len(intermediate_costs) == 0)

    def test_stores_the_params_in_the_model(self):
        intermediate_params = []

        generator = self.testmodel.learn_generator(self.X, self.Y, self.iterations, self.interval)

        for (_, params, _) in generator:
            intermediate_params.append(params)

        self.assertTrue(np.array_equal(intermediate_params[-1]['W1'], self.testmodel.params['W1']))

    def test_throws_error_when_interval_is_less_than_0(self):
        pass

    def test_throws_error_when_interval_is_greater_than_max_iterations(self):
        pass

class DeepenModelLearnTest(unittest.TestCase):
    def setUp(self):
        self.testmodel = model.Model()
        self.X = np.ones((1,1))
        self.Y = np.ones((1,1))
        self.iterations = 100

    def test_compiles_a_list_of_costs_from_all_iterations(self):
        intermediates = self.testmodel.learn(self.X, self.Y, self.iterations)

        self.assertTrue(len(intermediates) == self.iterations)
