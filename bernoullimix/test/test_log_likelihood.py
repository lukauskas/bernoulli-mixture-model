import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from pandas.util.testing import assert_series_equal, assert_frame_equal
from scipy.special import gammaln

import pandas as pd

from bernoullimix.mixture import MultiDatasetMixtureModel, _responsibilities_from_log_support


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

        ms_one = pd.Series([0.1, 0.5, 0.4], index=es.index)

        mixture = MultiDatasetMixtureModel(ms_one, es)

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

        ms_one = pd.Series([0.1, 0.5, 0.4], index=es.index)

        mixture = MultiDatasetMixtureModel(ms_one, es)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset)

    def test_log_likelihood_validates_dataset_id_index(self):
        """
        log likelihood function should validate that dataset index corresponds to the one provided
        """
        dataset = pd.DataFrame([[False, True, False, 'x', 1],
                                [False, False, False, 'y', 1]],
                                columns=['a', 'b', 'c', 'dataset_id', 'weight'])

        es = pd.DataFrame([[0.1, 0.2, 0.6],
                           [0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4]],
                          columns=['a', 'b', 'c'])

        ms_two = pd.DataFrame([[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]],
                              columns=es.index,
                              index=['a', 'b'])

        mixture = MultiDatasetMixtureModel(ms_two, es)

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

        ms_two = pd.DataFrame([[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]],
                              columns=es.index,
                              index=['x', 'y'])

        mixture = MultiDatasetMixtureModel(ms_two, es)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_weight_zero)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_weight_nan)

        self.assertRaises(ValueError, mixture.log_likelihood, dataset_weight_negative)

    def test_individual_log_likelihood(self):

        row = pd.DataFrame([[True, False, None, 'dataset-a', 2.5]],
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

        model = MultiDatasetMixtureModel(pi, p)

        expected_log_likelihood = pd.Series([2.5 * (np.log(
            sum([pi.loc['dataset-a', k] * p.loc[k, 'X1'] * (1-p.loc[k, 'X2']) for k in ['K0', 'K1']])
        ))], index=row.index)

        actual_log_likelihood = model._individual_log_likelihoods(row)
        assert_series_equal(actual_log_likelihood, expected_log_likelihood, check_names=False)

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

        model = MultiDatasetMixtureModel(pi, p)

        individual_lls = model._individual_log_likelihoods(sample_data)
        expected_log_likelihood = individual_lls.sum()

        actual_log_likelihood = model.log_likelihood(sample_data)

        self.assertEqual(expected_log_likelihood, actual_log_likelihood)

    def test_posterior_calculation_from_log_likelihood(self):

            pi = pd.DataFrame([[0.6, 0.4],
                               [0.2, 0.8],
                               [0.5, 0.5]],
                              index=['dataset-a', 'dataset-b', 'dataset-c'],
                              columns=['K0', 'K1'])

            pi_priors = pd.Series([4, 5], index=pi.columns)

            p = pd.DataFrame([[0.1, 0.2, 0.3],
                              [0.9, 0.8, 0.7]],
                             index=['K0', 'K1'],
                             columns=['X1', 'X2', 'X3'])

            p_priors = pd.DataFrame([[6, 7],
                                     [8, 9],
                                     [10, 11]], index=p.columns, columns=['alpha', 'beta'])

            model = MultiDatasetMixtureModel(pi, p, prior_mixing_coefficients=pi_priors,
                                             prior_emission_probabilities=p_priors)

            log_likelihood = np.log(1e-5)

            pi_prior_weight = (pi.apply(np.log) * (pi_priors - 1)).sum().sum()


            pi_prior_gammas = len(pi.index) * (gammaln(pi_priors.sum()) - pi_priors.apply(gammaln).sum())

            p_prior_weight = (p.apply(np.log) * (p_priors['alpha'] - 1) + (1-p).apply(np.log) * (p_priors['beta'] - 1)).sum().sum()


            p_prior_gammas = len(p.index) * (p_priors.sum(axis=1).apply(gammaln)
                                             - p_priors['alpha'].apply(gammaln)
                                             - p_priors['beta'].apply(gammaln)).sum()

            expected_posterior_no_gammas = log_likelihood + pi_prior_weight + p_prior_weight

            expected_posterior_with_gammas = expected_posterior_no_gammas \
                                             + pi_prior_gammas + p_prior_gammas

            posterior_no_gammas = model._unnormalised_posterior(log_likelihood,
                                                                compute_gammas=False)
            posterior_gammas = model._unnormalised_posterior(log_likelihood,
                                                             compute_gammas=True)

            self.assertAlmostEqual(expected_posterior_no_gammas, posterior_no_gammas)
            self.assertAlmostEqual(expected_posterior_with_gammas, posterior_gammas)


    def test_support(self):
        row = pd.DataFrame([[True, False, None, 'dataset-a', 2.5]],
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

        model = MultiDatasetMixtureModel(pi, p)

        expected_support = pd.DataFrame([[
            pi.loc['dataset-a', 'K0'] * p.loc['K0', 'X1'] * (1 - p.loc['K0', 'X2']),
            pi.loc['dataset-a', 'K1'] * p.loc['K1', 'X1'] * (1 - p.loc['K1', 'X2']),
        ]], columns=pi.columns, index=row.index)

        expected_log_support = expected_support.apply(np.log)
        dataset_ids_as_ilocs = model._dataset_ids_as_pis_ilocs(row)

        actual_log_support = model._log_support(dataset_ids_as_ilocs, *model._to_bool(row))

        assert_frame_equal(expected_log_support, actual_log_support)

    def test_responsibilities_from_log_support(self):
        data = pd.DataFrame([[True, False, None, 'dataset-a', 2.5],
                            [False, True, True, 'dataset-b', 15],
                            [False, False, False, 'dataset-c', 20]],
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

        model = MultiDatasetMixtureModel(pi, p)

        dataset_ids_as_ilocs = model._dataset_ids_as_pis_ilocs(data)
        log_support = model._log_support(dataset_ids_as_ilocs, *model._to_bool(data))
        support = np.exp(log_support)

        expected_responsibilities = support.divide(support.sum(axis=1), axis=0)
        actual_responsibilities = _responsibilities_from_log_support(log_support)

        assert_frame_equal(expected_responsibilities, actual_responsibilities)


    def test_pi_update_from_data(self):

        sample_data = pd.DataFrame([[True, True, None, 'dataset-a', 2.5],
                                    [False, None, False, 'dataset-b', 1.5],
                                    [True, False, True, 'dataset-a', 5],
                                    [False, False, True, 'dataset-c', 1]],
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

        model = MultiDatasetMixtureModel(pi, p)

        zstar = pd.DataFrame([[0.9, 0.1],
                              [0.1, 0.9],
                              [0.5, 0.5],
                              [0.4, 0.6]],
                             index=sample_data.index,
                             columns=pi.columns)

        pi_expected = pd.DataFrame([
            [(2.5 * 0.9 + 5 * 0.5), (2.5 * 0.1 + 5 * 0.5)],
            [(1.5 * 0.1), (1.5 * 0.9)],
            [(1 * 0.4), (1 * 0.6)]
        ],
            index=pi.index,
            columns=pi.columns)

        denominators = pd.DataFrame([
            [2.5 + 5, 2.5 + 5],
            [1.5, 1.5],
            [1, 1]
        ], index=pi.index, columns=pi.columns)

        pi_expected /= denominators

        pi_actual = model._pi_update_from_data(sample_data, zstar)
        assert_frame_equal(pi_expected, pi_actual)

    def test_pi_update_with_priors(self):
        sample_data = pd.DataFrame([[True, True, None, 'dataset-a', 2.5],
                                    [False, None, False, 'dataset-b', 1.5],
                                    [True, False, True, 'dataset-a', 5],
                                    [False, False, True, 'dataset-c', 1]],
                                   columns=['X1', 'X2', 'X3', 'dataset_id', 'weight'])

        pi = pd.DataFrame([[0.6, 0.4],
                           [0.2, 0.8],
                           [0.5, 0.5]],
                          index=['dataset-a', 'dataset-b', 'dataset-c'],
                          columns=['K0', 'K1'])

        pi_prior = pd.Series([2, 3], index=pi.columns)

        p = pd.DataFrame([[0.1, 0.2, 0.3],
                          [0.9, 0.8, 0.7]],
                         index=['K0', 'K1'],
                         columns=['X1', 'X2', 'X3'])

        model = MultiDatasetMixtureModel(pi, p, prior_mixing_coefficients=pi_prior)

        zstar = pd.DataFrame([[0.9, 0.1],
                              [0.1, 0.9],
                              [0.5, 0.5],
                              [0.4, 0.6]],
                             index=sample_data.index,
                             columns=pi.columns)

        pi_expected = pd.DataFrame([
            [(2.5 * 0.9 + 5 * 0.5), (2.5 * 0.1 + 5 * 0.5)],
            [(1.5 * 0.1), (1.5 * 0.9)],
            [(1 * 0.4), (1 * 0.6)]
        ],
            index=pi.index,
            columns=pi.columns)

        pi_expected += pi_prior - 1

        denominators = pd.DataFrame([
            [2.5 + 5, 2.5 + 5],
            [1.5, 1.5],
            [1, 1]
        ], index=pi.index, columns=pi.columns)

        denominators += pi_prior.sum() - len(pi_prior)

        pi_expected /= denominators

        print(pi_expected)

        pi_actual = model._pi_update_from_data(sample_data, zstar)
        assert_frame_equal(pi_expected, pi_actual)

    def test_dataset_collapse(self):
        sample_data = pd.DataFrame([[True, True, None, 'dataset-a', 2.5],   # A
                                    [False, None, False, 'dataset-b', 1.5],  # B
                                    [True, True, None, 'dataset-a', 3.5],  # A
                                    [True, False, True, 'dataset-a', 5],  # C
                                    [False, False, True, 'dataset-c', 1],  # D
                                    [True, False, True, 'dataset-a', 3.5],  # C
                                    [True, False, True, 'dataset-a', 2],  # C
                                    [False, False, True, 'dataset-c', 1.5],  # D
                                    [None, None, True, 'dataset-c', 1.5],  # E,
                                    [None, None, True, 'dataset-b', 1],  # F,
                                    [None, None, True, 'dataset-c', 2.5],  # E,
                                    [None, None, True, 'dataset-b', 2.1],  # F,
                                    [True, True, np.nan, 'dataset-a', 15],  # A (with nan)
                                    ],
                                   columns=['X1', 'X2', 'X3', 'dataset_id', 'weight'])

        expected_collapsed = pd.DataFrame([[True, True, None, 'dataset-a', 2.5 + 3.5 + 15],   # A
                                    [False, None, False, 'dataset-b', 1.5],  # B
                                    [True, False, True, 'dataset-a', 5 + 3.5 + 2],  # C
                                    [False, False, True, 'dataset-c', 1 + 1.5],  # D
                                    [None, None, True, 'dataset-c', 1.5 + 2.5],  # E,
                                    [None, None, True, 'dataset-b', 1 + 2.1],  # F,
                                    ],
                                   columns=['X1', 'X2', 'X3', 'dataset_id', 'weight'])

        expected_collapsed.sort_values(by='weight', inplace=True)
        # sorting messes up indices
        expected_collapsed.index = range(len(expected_collapsed))

        actual_collapsed = MultiDatasetMixtureModel.collapse_dataset(sample_data)
        actual_collapsed.sort_values(by='weight', inplace=True)
        actual_collapsed.index = range(len(actual_collapsed))  # Fix sorting index

        print('Expected')
        print(expected_collapsed)
        print('\nActual:')
        print(actual_collapsed)
        print()
        assert_frame_equal(expected_collapsed, actual_collapsed)

    def test_p_update_from_data(self):

        sample_data = pd.DataFrame([[True, True, None, 'dataset-a', 2.5],
                                    [False, None, False, 'dataset-b', 1.5],
                                    [True, False, True, 'dataset-a', 5],
                                    [False, False, True, 'dataset-c', 1]],
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

        model = MultiDatasetMixtureModel(pi, p)

        zstar = pd.DataFrame([[0.9, 0.1],
                              [0.1, 0.9],
                              [0.5, 0.5],
                              [0.4, 0.6]],
                             index=sample_data.index,
                             columns=pi.columns)

        denominators = pd.Series([0.9 * 2.5 + 0.1 * 1.5 + 0.5 * 5 + 0.4 * 1,
                                  0.1 * 2.5 + 0.9 * 1.5 + 0.5 * 5 + 0.6 * 1],
                                 index=p.index)

        denominators = pd.concat([denominators] * len(p.columns), axis=1)
        denominators.columns = p.columns

        expected_p = pd.DataFrame(
            [np.array([2.5 * 0.9 + 5 * 0.5, 2.5 * 0.9 + 1.5 * 0.1 * 0.2,
                       2.5 * 0.9 * 0.3 + 5 * 0.5 + 1 * 0.4]),
             np.array([2.5 * 0.1 + 5 * 0.5, 2.5 * 0.1 + 1.5 * 0.9 * 0.8,
                       2.5 * 0.1 * 0.7 + 5 * 0.5 + 1 * 0.6])],
            index=p.index,
            columns=p.columns,
        )

        expected_p /= denominators

        data_as_bool, not_null_mask = model._to_bool(sample_data)

        zstar_times_weight = zstar.multiply(sample_data['weight'], axis=0)

        actual_p = model._p_update_from_data(zstar_times_weight.values,
                                             data_as_bool, not_null_mask)

        assert_frame_equal(expected_p, actual_p)

    def test_p_update_with_priors(self):
        sample_data = pd.DataFrame([[True, True, None, 'dataset-a', 2.5],
                                    [False, None, False, 'dataset-b', 1.5],
                                    [True, False, True, 'dataset-a', 5],
                                    [False, False, True, 'dataset-c', 1]],
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

        p_priors = pd.DataFrame([[1, 2],
                                 [3, 4],
                                 [5, 6]], index=p.columns, columns=['alpha', 'beta'])

        model = MultiDatasetMixtureModel(pi, p, prior_emission_probabilities=p_priors)

        zstar = pd.DataFrame([[0.9, 0.1],
                              [0.1, 0.9],
                              [0.5, 0.5],
                              [0.4, 0.6]],
                             index=sample_data.index,
                             columns=pi.columns)

        denominators = pd.Series([0.9 * 2.5 + 0.1 * 1.5 + 0.5 * 5 + 0.4 * 1,
                                  0.1 * 2.5 + 0.9 * 1.5 + 0.5 * 5 + 0.6 * 1],
                                 index=p.index)

        denominators = pd.concat([denominators]*len(p.columns), axis=1)
        denominators.columns = p.columns

        expected_p = pd.DataFrame(
            [np.array([2.5 * 0.9 + 5 * 0.5, 2.5 * 0.9 + 1.5 * 0.1 * 0.2,
                       2.5 * 0.9 * 0.3 + 5 * 0.5 + 1 * 0.4]),
             np.array([2.5 * 0.1 + 5 * 0.5, 2.5 * 0.1 + 1.5 * 0.9 * 0.8,
                       2.5 * 0.1 * 0.7 + 5 * 0.5 + 1 * 0.6])],
            index=p.index,
            columns=p.columns,
        )

        expected_p += p_priors['alpha'] - 1
        denominators += p_priors['alpha'] + p_priors['beta'] - 2

        expected_p /= denominators

        data_as_bool, not_null_mask = model._to_bool(sample_data)
        zstar_times_weight = zstar.multiply(sample_data['weight'], axis=0)

        actual_p = model._p_update_from_data(zstar_times_weight.values,
                                             data_as_bool, not_null_mask)

        assert_frame_equal(expected_p, actual_p)