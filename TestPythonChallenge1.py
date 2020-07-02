import unittest
import numpy as np
from PythonChallenge1 import Community

class TestProcessing(unittest.TestCase):

    def setUp(self):
        self.community = Community(100, 8, 0.05)

    def test_initial_population(self):
        self.assertEqual(self.community.population_array.shape, (100, 8))
        n = set(np.arange(8))

        for i in list(set(self.community.population_array.flatten())):
            self.assertIn(i, n)

    def test_fitness_score(self):
        self.assertLessEqual(max(self.community.score), 8)
        self.assertGreaterEqual(min(self.community.score), 0)

    def test_get_score(self):
        self.assertEqual(self.community.population_array.shape, (100, 8))
        n = set(np.arange(8))

        for i in list(set(self.community.get_score())):
            self.assertIn(i, n)

    def test_crossover_mutation(self):
        old_generation = self.community.population_array
        self.community.crossover_mutation()
        self.assertRaises(AssertionError, np.testing.assert_array_equal, old_generation,
                          self.community.population_array)
        self.assertEqual(self.community.population_array.shape, (100, 8))
        n = set(np.arange(8))

        for i in list(set(self.community.population_array.flatten())):
            self.assertIn(i, n)

    def test_find_perfect_config(self):
        self.community.population_array = np.zeros((100, 8))
        self.community.population_array[0] = np.arange(8)
        self.community.fitness_score()
        self.community.find_perfect_config()
        np.testing.assert_array_equal(self.community.perfect_config, np.arange(8))

    def test_get_perfect_config(self):
        self.community.perfect_config = np.arange(8)
        self.assertEqual(self.community.get_perfect_config(), '0 1 2 3 4 5 6 7')
