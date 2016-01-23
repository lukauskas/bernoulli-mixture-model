import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from pandas.util.testing import assert_series_equal

from bernoullimix import BernoulliMixture
import pandas as pd

from bernoullimix.mixture import MultiDatasetMixtureModel


class TestLogLikelihoodNew(unittest.TestCase):

    def test_log_likelihood_checks_that_dataset_id_and_weight_columns_exist(self):
        """
        log likelihood function should validate that data has dataset_id and weight columns
        :return:
        """

        dataset_no_weight = pd.DataFrame([[False, True, False, 0],
                                          [False, False, False, 0]],
                                          columns=['a', 'b', 'c', 'dataset_id'])

        dataset_no_dataset_id = pd.DataFrame([[False, True, False, 1],
                                          [False, False, False, 1]],
                                          columns=['a', 'b', 'c', 'weight'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=['a', 'b', 'c'])

        dataset_mu = pd.Series([1])
        ms_one = pd.Series([0.1, 0.5, 0.4], index=es.index)

        mixture = MultiDatasetMixtureModel(dataset_mu, ms_one, es)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_no_weight)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_no_dataset_id)

    def test_log_likelihood_validates_all_data_columns_present(self):
        """
        log likelihood function should validate that data provided has all the data columns
        """
        dataset = pd.DataFrame([[False, True, False, 0, 1],
                                [False, False, False, 0, 1]],
                                columns=['a', 'b', 'c', 'dataset_id', 'weight'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=['a', 'b', 'd'])  # column d not in data

        dataset_mu = pd.Series([1])
        ms_one = pd.Series([0.1, 0.5, 0.4], index=es.index)

        mixture = MultiDatasetMixtureModel(dataset_mu, ms_one, es)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset)

    def test_log_likelihood_validates_dataset_id_index(self):
        """
        log likelihood function should validate that dataset index corresponds to the one provided
        """
        dataset = pd.DataFrame([[False, True, False, 'x', 1],
                                [False, False, False, 'y', 1]],
                                columns=['a', 'b', 'c', 'dataset_id', 'weight'])

        dataset_mu = pd.Series([0.1, 0.9], index=['a', 'b'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=['a', 'b', 'c'])

        ms_two = pd.DataFrame([[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]],
                              columns=es.index,
                              index=['a', 'b'])

        mixture = MultiDatasetMixtureModel(dataset_mu, ms_two, es)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset)

    def test_mixture_validates_dataset_mu_index_on_init(self):

        dataset_mu = pd.Series([0.1, 0.9], index=['x', 'z'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=['a', 'b', 'c'])

        ms_two = pd.DataFrame([[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]],
                              columns=es.index,
                              index=['x', 'y'])

        self.assertRaises(ValueError, MultiDatasetMixtureModel, dataset_mu, ms_two, es)

    def test_log_likelihood_validates_weights_greater_than_zero(self):
        """
        log likelihood function should validate that dataset weights are > 0
        """
        dataset_weight_zero = pd.DataFrame([[False, True, False, 'x', 1],
                                           [False, False, False, 'y', 0]],
                                           columns=['a', 'b', 'c', 'dataset_id', 'weight'])

        dataset_weight_nan = pd.DataFrame([[False, True, False, 'x', None],
                                           [False, False, False, 'y', 1.0]],
                                           columns=['a', 'b', 'c', 'dataset_id', 'weight'])

        dataset_weight_negative = pd.DataFrame([[False, True, False, 'x', -1],
                                           [False, False, False, 'y', 1.0]],
                                           columns=['a', 'b', 'c', 'dataset_id', 'weight'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=['a', 'b', 'c'])

        dataset_mu = pd.Series([0.1, 0.9], index=['x', 'y'])

        ms_two = pd.DataFrame([[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]],
                              columns=es.index,
                              index=['x', 'y'])

        mixture = MultiDatasetMixtureModel(dataset_mu, ms_two, es)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_weight_zero)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_weight_nan)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_weight_negative)

    def test_log_likelihood_for_row(self):

        row = pd.Series([True, False, None, 'dataset-a', 2.5],
                         index=['X1', 'X2', 'X3', 'dataset_id', 'weight'])

        pi = pd.DataFrame([[0.6, 0.4],
                           [0.2, 0.8],
                           [0.5, 0.5]],
                          index=['dataset-a', 'dataset-b', 'dataset-c'],
                          columns=['K0', 'K1'])

        p = pd.DataFrame([[0.1, 0.2, 0.3],
                          [0.9, 0.8, 0.7]],
                         index=['K0', 'K1'],
                         columns=['X1', 'X2', 'X3'])

        mu = pd.Series([0.5, 0.25, 0.25], index=['dataset-a', 'dataset-b', 'dataset-c'])

        model = MultiDatasetMixtureModel(mu, pi, p)

        expected_log_likelihood = np.log(mu.loc['dataset-a']) + np.log(
            sum([pi.loc['dataset-a', k] * p.loc[k, 'X1'] * (1-p.loc[k, 'X2']) for k in ['K0', 'K1']])
        )

        actual_log_likelihood = model._log_likelihood_for_row(row)

        self.assertEqual(expected_log_likelihood, actual_log_likelihood)


    def test_log_likelihood_weighs_data_correctly(self):

        sample_data = pd.DataFrame([[True, True, None, 'dataset-a', 2.5],
                                    [False, None, False, 'dataset-b', 1],
                                    [True, False, True, 'dataset-a', 10],
                                    [False, False, True, 'dataset-c', 3]],
                                   columns=['X1', 'X2', 'X3', 'dataset_id', 'weight'])

        pi = pd.DataFrame([[0.6, 0.4],
                           [0.2, 0.8],
                           [0.5, 0.5]],
                          index=['dataset-a', 'dataset-b', 'dataset-c'],
                          columns=['K0', 'K1'])

        p = pd.DataFrame([[0.1, 0.2, 0.3],
                          [0.9, 0.8, 0.7]],
                         index=['K0', 'K1'],
                         columns=['X1', 'X2', 'X3'])

        mu = pd.Series([0.5, 0.25, 0.25], index=['dataset-a', 'dataset-b', 'dataset-c'])

        model = MultiDatasetMixtureModel(mu, pi, p)

        individual_lls = sample_data.apply(model._log_likelihood_for_row, axis=1)

        expected_log_likelihood = (individual_lls * sample_data['weight']).sum()
        actual_log_likelihood = model.log_likelihood(sample_data)

        self.assertEqual(expected_log_likelihood, actual_log_likelihood)

    def test_support(self):

        row = pd.Series([True, False, None, 'dataset-a', 2.5],
                         index=['X1', 'X2', 'X3', 'dataset_id', 'weight'])

        pi = pd.DataFrame([[0.6, 0.4],
                           [0.2, 0.8],
                           [0.5, 0.5]],
                          index=['dataset-a', 'dataset-b', 'dataset-c'],
                          columns=['K0', 'K1'])

        p = pd.DataFrame([[0.1, 0.2, 0.3],
                          [0.9, 0.8, 0.7]],
                         index=['K0', 'K1'],
                         columns=['X1', 'X2', 'X3'])

        mu = pd.Series([0.5, 0.25, 0.25], index=['dataset-a', 'dataset-b', 'dataset-c'])

        model = MultiDatasetMixtureModel(mu, pi, p)

        expected_support = pd.Series([
            pi.loc['dataset-a', 'K0'] * p.loc['K0', 'X1'] * (1 - p.loc['K0', 'X2']),
            pi.loc['dataset-a', 'K1'] * p.loc['K1', 'X1'] * (1 - p.loc['K1', 'X2']),
        ], index=pi.columns)

        actual_support = model._support_for_row(row)

        assert_series_equal(expected_support, actual_support)





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