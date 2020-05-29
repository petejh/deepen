import io
import unittest

import h5py as h5
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

class DeepenModelSaveTest(unittest.TestCase):
    def setUp(self):
        self.testmodel = model.Model()
        self.datafile = io.BytesIO()

    def test_saves_all_properties(self):
        self.testmodel.save(self.datafile)

        with h5.File(self.datafile, 'r') as df:
            for prop in ["learning_rate", "layer_dims", "params"]:
                with self.subTest(prop = prop):
                    self.assertTrue(prop in df)

    def test_saves_the_learning_rate(self):
        self.testmodel.save(self.datafile)

        with h5.File(self.datafile, 'r') as df:
            saved_rate = df["learning_rate"][()].item()

        self.assertTrue(saved_rate == self.testmodel.learning_rate)

    def tearDown(self):
        self.datafile.close()

class DeepenModelLoadTest(unittest.TestCase):
    def setUp(self):
        self.testmodel = model.Model()

        self.learning_rate = 0.08
        self.layer_dims = [1, 10, 1]
        self.params = {
            'W1': np.ones((10, 1)),
            'b1': np.ones((10, 1))
        }
        self.datafile = io.BytesIO()

        with h5.File(self.datafile, 'w') as df:
            df.create_dataset("learning_rate", data=self.learning_rate)
            df.create_dataset("layer_dims", data=self.layer_dims)
            df.create_dataset("params/W1", data=self.params['W1'])
            df.create_dataset("params/b1", data=self.params['b1'])

    def test_loads_layer_dims_as_a_native_type(self):
        # h5py loads data in numpy formats by default; we want native types
        self.testmodel.load(self.datafile)

        self.assertIs(type(self.testmodel.layer_dims), list)

    def test_loads_learning_rate_as_a_native_type(self):
        # h5py loads data in numpy formats by default; we want native types
        self.testmodel.load(self.datafile)

        self.assertIs(type(self.testmodel.learning_rate), float)

    def test_loads_the_learning_rate(self):
        self.testmodel.load(self.datafile)

        self.assertTrue(self.testmodel.learning_rate == self.learning_rate)

    def test_loads_all_params(self):
        self.testmodel.load(self.datafile)

        self.assertTrue(len(self.testmodel.params) == len(self.params))

    def tearDown(self):
        self.datafile.close()
