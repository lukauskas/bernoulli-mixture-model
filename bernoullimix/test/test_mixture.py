from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from bernoullimix.mixture import BernoulliMixture

class TestInitialisation(unittest.TestCase):

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
        self.assertIsInstance(mixture.emission_probabilities, np.ndarray)
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

    def test_constant_initialisation_when_mixing_coeffiecients_do_not_sum_to_one(self):
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

    def test_constant_initialisation_when_mixing_coeffiecients_not_between_0_and_1(self):
        """
        Given wrong number of mixing components, initialiser should raise an error.
        """

        number_of_components = 3
        number_of_dimensions = 4

        less_than_zero = np.array([-0.5, 1, 0.5])
        more_than_one = np.array([1.5, -1.2, 0.7])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          less_than_zero, sample_emission_probabilities)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          more_than_one, sample_emission_probabilities)

    def test_constant_initialisation_when_emission_probabilities_are_bounded_appropriately(self):
        """
        Given that emission probabilities are greater than one or lower than one, raise error.
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        more_than_one = np.array([[0.1, 0.2, 0.3, 0.4],
                                  [0.1, 0.4, 0.1, 0.4],
                                  [1.0, 0.0, 1.8, 0.0]])

        less_than_zero = np.array([[0.1, 0.1, 0.3, 0.4],
                                  [0.1, 0.4, 0.1, 0.4],
                                  [1.0, 0.0, -5, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, more_than_one)
        self.assertRaises(ValueError, BernoulliMixture, number_of_components, number_of_dimensions,
                          sample_mixing_coefficients, less_than_zero)

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


class TestSampling(unittest.TestCase):

    def test_sampling_produces_appropriate_dimension_matrices(self):
        """
        Given K=3 components, D=4 dimensions and N=100 observations to generate,
        resulting matrices should have shapes (N,D) and (N,).
        :return:
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        size = 100
        observations, true_components = mixture.sample(size=size)

        self.assertEqual(observations.shape, (size, number_of_dimensions))
        self.assertEqual(true_components.shape, (size, ))

    def test_sampling_random_state(self):
        """
        Two samples with the same random state should be identical.
        Two samples with different random states should be different
        :return:
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        size = 100

        random_state_a = random_state_b = 123
        random_state_c = 321

        observations_a, true_components_a = mixture.sample(size=size, random_state=random_state_a)
        observations_b, true_components_b = mixture.sample(size=size, random_state=random_state_b)
        observations_c, true_components_c = mixture.sample(size=size, random_state=random_state_c)

        assert_array_equal(observations_a, observations_b)
        assert_array_equal(true_components_a, true_components_b)

        self.assertNotEqual(observations_a, observations_c)
        self.assertNotEqual(true_components_a, true_components_c)

    def test_sampling_generates_components_with_approximately_appropriate_probabilities(self):
        """
        Given three components and their associated emission matrices and 10000 samples,
        the generated dataset should reflect these probabilities approximately.
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.6, 0.3, 0.1])
        sample_emission_probabilities = np.array([[0.5, 0.3, 0.1, 0.9],
                                                  [0.0, 1.0, 0.0, 1.0],
                                                  [0.3, 0.3, 0.3, 0.3]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        size = 10000

        observations, true_components = mixture.sample(size=size, random_state=12345)

        component_counts = np.empty(number_of_components)
        component_observation_counts = np.empty((number_of_components, number_of_dimensions))

        for component in range(number_of_components):
            mask = true_components == component
            # noinspection PyTypeChecker
            component_counts[component] = np.sum(mask)

            component_observations = observations[mask]
            component_observation_counts[component] = np.sum(component_observations, axis=0)

        empirical_mixing_coefficients = component_counts / size
        empirical_emission_probabilities = component_observation_counts / size

        assert_array_almost_equal(empirical_mixing_coefficients, sample_mixing_coefficients)
        assert_array_almost_equal(empirical_emission_probabilities, sample_emission_probabilities)

