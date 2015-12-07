import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from bernoullimix import BernoulliMixture


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
                                                  [0.95, 0.05, 0.05, 0.05]])

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
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        expected_answer = np.empty((len(sample_dataset), number_of_components))

        for sample in range(len(sample_dataset)):
            for component in range(number_of_components):
                expected_answer[sample, component] = sample_mixing_coefficients[component] * \
                    np.product(np.power(sample_emission_probabilities[component], sample_dataset[sample]) *
                               np.power(1-sample_emission_probabilities[component], 1-sample_dataset[sample]))

        actual_answer = mixture._prob_z_o_given_theta(sample_dataset)

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
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        unique_dataset, weights = BernoulliMixture.aggregate_dataset(sample_dataset)

        # Compute support on whole dataset for the test
        # even though code would compute it for unique_dataset only
        support = mixture._prob_z_o_given_theta(sample_dataset)
        expected_answer = np.sum(np.log(np.sum(support, axis=1)))

        actual_answer = mixture.log_likelihood(sample_dataset)

        unique_support = mixture._prob_z_o_given_theta(unique_dataset)
        actual_answer_from_support = mixture._log_likelihood_from_support(unique_support, weights)

        self.assertAlmostEqual(expected_answer, actual_answer)
        self.assertAlmostEqual(expected_answer, actual_answer_from_support)