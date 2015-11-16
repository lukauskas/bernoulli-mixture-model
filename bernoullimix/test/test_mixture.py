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

class TestLogLikelihood(unittest.TestCase):

    def test_log_likelihood_validates_dataset_shape(self):
        """
        Given a dataset that has either too few or too many dimensions,
        log_likelihood function should raise an error message.

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

        dataset_too_many_dims = np.ones((10, 5))
        dataset_too_few_dims = np.ones((10, 3))

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_too_few_dims)
        self.assertRaises(ValueError, mixture.log_likelihood, dataset_too_many_dims)


    def test_probability_of_generating_a_point_computation(self):
        """
        Given a set of parameters for the model.
        The _point_emission_probs should return a K-sized vector X for each point where:
        $$
            x_k = p_k \prod_{d=1}^D \Theta_KD^{x_nd} (1 - \theta_Kd)^(1-x_nd)
        $$
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False]])

        expected_answer = np.empty((len(sample_dataset), number_of_components))

        for sample in range(len(sample_dataset)):
            for component in range(number_of_components):
                expected_answer[sample, component] = sample_mixing_coefficients[component] * \
                    np.product(np.power(sample_emission_probabilities[component], sample_dataset[sample]) *
                               np.power(1-sample_emission_probabilities[component], 1-sample_dataset[sample]))

        actual_answer = mixture._observation_emission_support(sample_dataset)

        assert_array_almost_equal(expected_answer, actual_answer)

    def test_log_likelihood_produces_correct_answer(self):
        """
        Given a set of parameters, and an appropriate dataset, log likelihood should
        return:

        $$
           \sum_n log \sum_k p_k \prod_{d=1}^D \Theta_KD^{x_nd} (1 - \theta_Kd)^(1-x_nd)
        $$
        """

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False]])

        support = mixture._observation_emission_support(sample_dataset)

        expected_answer = np.sum(np.log(np.sum(support, axis=1)))

        actual_answer = mixture.log_likelihood(sample_dataset)
        actual_answer_from_support = mixture._log_likelihood_from_support(support)

        self.assertEqual(expected_answer, actual_answer)
        self.assertEqual(expected_answer, actual_answer_from_support)
