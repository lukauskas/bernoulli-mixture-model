import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from bernoullimix import BernoulliMixture
import pandas as pd

from bernoullimix.mixture import MixtureModel


class TestLogLikelihoodNew(unittest.TestCase):

    def test_log_likelihood_validates_dataset_counts(self):
        """
        Given datasets, log likelihood should validate that appropriate amount of datasets is
        given (to match the number of priors)
        """

        dataset = pd.DataFrame([[False, True, False],
                                [False, False, False]],
                               columns=['a', 'b', 'c'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=dataset.columns)

        ms_one = pd.Series([0.1, 0.5, 0.4], index=es.index)
        ms_two = pd.DataFrame([[0.1, 0.5, 0.4],
                               [0.2, 0.2, 0.6]], columns=es.index)

        mixture_one_dataset = MixtureModel(ms_one, es)
        mixture_two_datasets = MixtureModel(ms_two, es)

        self.assertRaises(ValueError, mixture_one_dataset.log_likelihood, [dataset, dataset])
        self.assertRaises(ValueError, mixture_two_datasets.log_likelihood, [dataset])

    def test_log_likelihood_validates_dataset_dimensions(self):
        """
        Given datasets, mixture model should verify that their columns are the same as the
        emission probability columns.
        """

        dataset = pd.DataFrame([[False, True, False],
                                [False, False, False]],
                               columns=['a', 'b', 'c'])

        dataset_wrong_columns = pd.DataFrame([[False, True, False],
                                              [False, False, False]],
                                             columns=['a', 'b', 'e'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=dataset.columns)

        ms_two = pd.DataFrame([[0.1, 0.5, 0.4],
                               [0.2, 0.2, 0.6]], columns=es.index)

        mixture_two_datasets = MixtureModel(ms_two, es)

        self.assertRaises(ValueError, mixture_two_datasets.log_likelihood,
                          [dataset, dataset_wrong_columns])

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

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        dataset_too_many_dims = np.ones((10, 5))
        dataset_too_few_dims = np.ones((10, 3))

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_too_few_dims)
        self.assertRaises(ValueError, mixture.log_likelihood, dataset_too_many_dims)

    def test_computation_of_joint(self):

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

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

    def test_computation_of_joint_with_missing_data(self):
        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_eps = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_eps)

        sample_dataset = pd.DataFrame(np.array([[None, True, False, False],
                                                [False, None, False, False],
                                                [True, True, None, False],
                                                [False, True, False, None],
                                                [False, False, False, False],
                                                [None, None, None, None]]))

        expected_answer = np.empty((len(sample_dataset), number_of_components))

        for sample in range(len(sample_dataset)):
            for k in range(number_of_components):
                expected_answer[sample, k] = sample_mixing_coefficients[k]

                row = sample_dataset.iloc[sample]

                for d in range(len(row)):
                    if row[d] is None:
                        continue
                    expected_answer[sample, k] *= np.power(sample_eps[k, d], row[d])
                    expected_answer[sample, k] *= np.power(1-sample_eps[k, d], 1-row[d])

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

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset = pd.DataFrame(np.array([[None, True, False, False],
                                                [False, None, False, False],
                                                [True, True, None, False],
                                                [False, True, False, None],
                                                [False, False, False, False],
                                                [True, True, False, False]]))

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

    def test_log_likelihood_produces_correct_answer_with_missing_data(self):
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

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [None, None, None, None]])

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


class TestPenalisedLikelihood(unittest.TestCase):

    def test_number_of_free_parameters_computed_correctly(self):
        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        expected_number_of_free_parameters = (number_of_components - 1) + \
                                             (number_of_dimensions * number_of_components)
        actual_number_of_free_parameters = mixture.number_of_free_parameters

        self.assertEqual(actual_number_of_free_parameters, expected_number_of_free_parameters)

    def test_bic_computed_correctly(self):
        number_of_components = 3

        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        dataset = np.array([[True, True, False, False],
                            [False, True, False, False],
                            [True, True, False, False],
                            [False, True, False, False],
                            [False, False, False, False],
                            [True, True, False, False]])

        log_likelihood = mixture.log_likelihood(dataset)
        number_of_free_parameters = mixture.number_of_free_parameters

        expected_bic = -2 * log_likelihood + number_of_free_parameters * np.log(len(dataset))
        actual_bic = mixture.BIC_dataset(dataset)

        self.assertEqual(expected_bic, actual_bic)

    def test_bic_computed_correctly_with_missing_data(self):
        number_of_components = 3

        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        dataset = pd.DataFrame(np.array([[None, True, False, False],
                                         [False, None, False, False],
                                         [True, True, None, False],
                                         [False, True, False, None],
                                         [False, False, True, False],
                                         [None, None, None, None]]))

        log_likelihood = mixture.log_likelihood(dataset)
        number_of_free_parameters = mixture.number_of_free_parameters

        expected_bic = -2 * log_likelihood + number_of_free_parameters * np.log(len(dataset))
        actual_bic = mixture.BIC_dataset(dataset)

        self.assertEqual(expected_bic, actual_bic)

    def test_aic_computed_correctly(self):
        number_of_components = 3

        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        dataset = np.array([[True, True, False, False],
                            [False, True, False, False],
                            [True, True, False, False],
                            [False, True, False, False],
                            [False, False, False, False],
                            [True, True, False, False]])

        log_likelihood = mixture.log_likelihood(dataset)
        number_of_free_parameters = mixture.number_of_free_parameters

        expected_aic = -2 * log_likelihood + number_of_free_parameters * 2
        actual_aic = mixture.AIC_dataset(dataset)

        self.assertEqual(expected_aic, actual_aic)

    def test_aic_computed_correctly_with_missing_data(self):
        number_of_components = 3

        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

        dataset = pd.DataFrame(np.array([[None, True, False, False],
                                         [False, None, False, False],
                                         [True, True, None, False],
                                         [False, True, False, None],
                                         [False, False, True, False],
                                         [None, None, None, None]]))

        log_likelihood = mixture.log_likelihood(dataset)
        number_of_free_parameters = mixture.number_of_free_parameters

        expected_aic = -2 * log_likelihood + number_of_free_parameters * 2
        actual_aic = mixture.AIC_dataset(dataset)

        self.assertEqual(expected_aic, actual_aic)