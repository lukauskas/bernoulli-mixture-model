from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from bernoullimix.mixture import BernoulliMixture


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

        self.assertTrue(np.any(observations_a != observations_c))
        self.assertTrue(np.any(true_components_a != true_components_c))

    def test_sampling_generates_components_with_approximately_appropriate_probabilities(self):
        """
        Given three components and their associated emission matrices and 100000 samples,
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

        size = 100000

        observations, true_components = mixture.sample(size=size, random_state=12345)

        empirical_mixing_coefficients = np.empty(number_of_components)
        empirical_emission_probabilities = np.empty((number_of_components, number_of_dimensions))

        for component in range(number_of_components):
            mask = true_components == component
            # noinspection PyTypeChecker
            empirical_mixing_coefficients[component] = np.sum(mask) / size

            component_observations = observations[mask]
            empirical_emission_probabilities[component] = np.sum(component_observations, axis=0) / np.sum(mask)

        assert_array_almost_equal(empirical_mixing_coefficients, sample_mixing_coefficients,
                                  decimal=2)
        assert_array_almost_equal(empirical_emission_probabilities, sample_emission_probabilities,
                                  decimal=2)
