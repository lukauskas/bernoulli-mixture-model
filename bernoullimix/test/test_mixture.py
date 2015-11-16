from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from bernoullimix.mixture import BernoulliMixture

class TestInitialisation(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345)

    def test_constant_initialisation_with_array(self):
        """
        Given mixing coefficients and emission probabilities of appropriate dimensions
        BernoulliMixture should initialise just fine.
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)


        self.assertEqual(number_of_components, mixture.number_of_components)
        self.assertEqual(number_of_dimensions, mixture.number_of_dimensions)

        self.assertIsInstance(mixture.mixing_coefficients, np.ndarray)
        assert_array_equal(sample_mixing_coefficients, mixture.mixing_coefficients)
        self.assertIsInstance(sample_emission_probabilities, np.ndarray)
        assert_array_equal(sample_emission_probabilities, mixture.emission_probabilities)

    def test_constant_initialisation_with_list(self):
        """
        Given mixing coefficients and emission probabilities of appropriate dimensions
        in the form of python lists, BernoulliMixture should initialise just fine too.
        :return:
        """
        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = [0.5, 0.4, 0.1]
        sample_emission_probabilities = [[0.1, 0.2, 0.3, 0.4],
                                         [0.1, 0.4, 0.1, 0.4],
                                         [1.0, 0.0, 0.0, 0.0]]

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        self.assertEqual(number_of_components, mixture.number_of_components)
        self.assertEqual(number_of_dimensions, mixture.number_of_dimensions)

        self.assertIsInstance(mixture.mixing_coefficients, np.ndarray)
        assert_array_equal(sample_mixing_coefficients, mixture.mixing_coefficients)
        self.assertIsInstance(sample_emission_probabilities, np.ndarray)
        assert_array_equal(sample_emission_probabilities, mixture.emission_probabilities)

    def test_constant_initialisation_with_not_wrong_number_of_mixing_components(self):
        """
        Given wrong number of mixing components, initialiser should raise an error.
        """

        number_of_components = 3
        number_of_dimensions = 4

        too_few_components = np.array([0.5, 0.5])
        too_many_components = np.array([0.25, 0.25, 0.25, 0.25])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          too_few_components, sample_emission_probabilities)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          too_many_components, sample_emission_probabilities)

    def test_constant_initialisation_when_components_do_not_sum_to_one(self):
        """
        Given wrong number of mixing components, initialiser should raise an error.
        """

        number_of_components = 3
        number_of_dimensions = 4

        less_than_one = np.array([0.5, 0.4, 0.05])
        more_than_one = np.array([0.25, 0.25, 0.7])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          less_than_one, sample_emission_probabilities)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          more_than_one, sample_emission_probabilities)

    def test_constant_initialisation_when_emission_probabilities_do_not_sum_to_one(self):
        """
        Given that emission probabilities do not sum to one, initialiser should fail.
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        more_than_one = np.array([[0.1, 0.2, 0.3, 0.4],
                                  [0.1, 0.4, 0.1, 0.4],
                                  [1.0, 0.0, 0.8, 0.0]])

        less_than_one = np.array([[0.1, 0.1, 0.3, 0.4],
                                  [0.1, 0.4, 0.1, 0.4],
                                  [1.0, 0.0, 0.8, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, more_than_one)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, less_than_one)

    def test_constant_initialisation_wrong_emission_probabilities_dimension(self):
        """
        Given a wrong dimension of sample emission probabilities, the initialiser should fail
        with a value error.
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        emission_probabilities_too_few_components = np.array([[0.1, 0.2, 0.3, 0.4],
                                                              [1.0, 0.0, 0.0, 0.0]])
        emission_probabilities_too_many_components = np.array([[0.1, 0.2, 0.3, 0.4],
                                                               [1.0, 0.0, 0.0, 0.0],
                                                               [1.0, 0.0, 0.0, 0.0],
                                                               [1.0, 0.0, 0.0, 0.0]])

        emission_probabilities_too_many_dimensions = np.array([[0.1, 0.2, 0.3, 0.4, 0.0],
                                                               [1.0, 0.0, 0.0, 0.0, 0.0],
                                                               [1.0, 0.0, 0.0, 0.0, 0.0]])

        emission_probabilities_too_few_dimensions = np.array([[0.1, 0.2, 0.7],
                                                              [1.0, 0.0, 0.0],
                                                              [1.0, 0.0, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, emission_probabilities_too_few_components)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, emission_probabilities_too_many_components)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, emission_probabilities_too_few_dimensions)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, emission_probabilities_too_many_dimensions)