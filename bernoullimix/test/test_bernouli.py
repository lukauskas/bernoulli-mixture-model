import unittest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from bernoullimix import BernoulliMixture
from bernoullimix._bernoulli import probability_z_o_given_theta_c, bernoulli_prob_for_observations_with_mask, \
    _m_step, impute_missing_data_c

import pandas as pd

class TestBernoulliEmissionProbabilities(unittest.TestCase):

    def test_all_observed_no_mask(self):
        ps = np.array([0.3, 0.45])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        expected_answer = np.array([0.3 * 0.45, 0.3 * 0.55, 0.7 * 0.55, 0.7 * 0.45])

        actual_answer = bernoulli_prob_for_observations_with_mask(ps, data,
                                                                  np.ones(data.shape, dtype=bool))

        assert_array_almost_equal(actual_answer, expected_answer)

    def test_all_observed_explicit_mask(self):
        ps = np.array([0.3, 0.45])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        expected_answer = np.array([0.3 * 0.45, 0.3 * 0.55, 0.7 * 0.55, 0.7 * 0.45])

        mask = np.ones(data.shape, dtype=bool)

        actual_answer = bernoulli_prob_for_observations_with_mask(ps, data, mask)

        assert_array_almost_equal(actual_answer, expected_answer)

    def test_some_observed(self):
        ps = np.array([0.3, 0.45])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        mask = np.array([[True, False],
                         [True, True],
                         [False, True],
                         [True, False]])

        expected_answer = np.array([0.3, 0.3 * 0.55, 0.55, 0.7])

        actual_answer = bernoulli_prob_for_observations_with_mask(ps, data, mask)

        assert_array_almost_equal(actual_answer, expected_answer)

    def test_none_observed_results_to_one(self):
        ps = np.array([0.3, 0.45])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        mask = np.array([[False, False],
                         [False, False],
                         [False, False],
                         [False, False]])

        expected_answer = np.array([1, 1, 1, 1], dtype=np.float)

        actual_answer = bernoulli_prob_for_observations_with_mask(ps, data, mask)

        assert_array_almost_equal(actual_answer, expected_answer)

class TestBernoulliJoint(unittest.TestCase):

    def test_observed_joint_computation_all_observed_no_mask(self):

        pis = np.array([0.5, 0.25, 0.25])
        ps = np.array([[0.5, 0.3],
                       [0.3, 0.2],
                       [0.9, 0.1]])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        expected_answer = np.array([[pis[0] * 0.5 * 0.3, pis[1] * 0.3 * 0.2, pis[2] * 0.9 * 0.1],
                                    [pis[0] * 0.5 * 0.7, pis[1] * 0.3 * 0.8, pis[2] * 0.9 * 0.9],
                                    [pis[0] * 0.5 * 0.7, pis[1] * 0.7 * 0.8, pis[2] * 0.1 * 0.9],
                                    [pis[0] * 0.5 * 0.3, pis[1] * 0.7 * 0.2, pis[2] * 0.1 * 0.1]])

        actual_answer = probability_z_o_given_theta_c(data, np.ones(data.shape, dtype=bool), ps,
                                                      pis)

        assert_array_almost_equal(expected_answer, actual_answer)

    def test_observed_joint_computation_all_observed_mask(self):

        pis = np.array([0.5, 0.25, 0.25])
        ps = np.array([[0.5, 0.3],
                       [0.3, 0.2],
                       [0.9, 0.1]])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        expected_answer = np.array([[pis[0] * 0.5 * 0.3, pis[1] * 0.3 * 0.2, pis[2] * 0.9 * 0.1],
                                    [pis[0] * 0.5 * 0.7, pis[1] * 0.3 * 0.8, pis[2] * 0.9 * 0.9],
                                    [pis[0] * 0.5 * 0.7, pis[1] * 0.7 * 0.8, pis[2] * 0.1 * 0.9],
                                    [pis[0] * 0.5 * 0.3, pis[1] * 0.7 * 0.2, pis[2] * 0.1 * 0.1]])

        mask = np.ones(data.shape, dtype=bool)
        actual_answer = probability_z_o_given_theta_c(data, mask, ps, pis)

        assert_array_almost_equal(expected_answer, actual_answer)

    def test_observed_joint_computation_some_observed_mask(self):

        pis = np.array([0.5, 0.25, 0.25])
        ps = np.array([[0.5, 0.3],
                       [0.3, 0.2],
                       [0.9, 0.1]])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        # Unobserved values should not impact anything, so they can be flipped
        data2 = np.array([[True, True],
                         [False, False],
                         [False, True],
                         [True, False]])

        expected_answer = np.array([[pis[0] * 0.5 * 0.3, pis[1] * 0.3 * 0.2, pis[2] * 0.9 * 0.1],
                                    [pis[0] * 0.7, pis[1] * 0.8, pis[2] * 0.9],
                                    [pis[0] * 0.5, pis[1] * 0.7, pis[2] * 0.1],
                                    [pis[0], pis[1], pis[2]]])

        mask = np.array([[True, True],
                         [False, True],
                         [True, False],
                         [False, False]])

        actual_answer = probability_z_o_given_theta_c(data, mask, ps, pis)
        actual_answer2 = probability_z_o_given_theta_c(data2, mask, ps, pis)

        assert_array_almost_equal(expected_answer, actual_answer)
        assert_array_almost_equal(expected_answer, actual_answer2)


class TestMStep(unittest.TestCase):
    def test_m_step_computes_correct_parameters_no_mask(self):
        sample_z_star = np.array([[0.24522862, 0.50447031, 0.25030106],
                                  [0.3264654, 0.67158596, 0.00194864],
                                  [0.24522862, 0.50447031, 0.25030106],
                                  [0.3264654, 0.67158596, 0.00194864],
                                  [0.55562318, 0.4286236, 0.01575322],
                                  [0.24522862, 0.50447031, 0.25030106]])

        unique_z_star = np.array([[0.24522862, 0.50447031, 0.25030106],
                                  [0.3264654, 0.67158596, 0.00194864],
                                  [0.55562318, 0.4286236, 0.01575322]])

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        unique_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   ])

        weights = np.array([3, 2, 1], dtype=int)

        # -- Compute correct parameters from the full dataset

        N, D = sample_dataset.shape
        __, K = sample_z_star.shape

        u = np.sum(sample_z_star, axis=0)

        expected_mixing_coefficients = u / N

        expected_emission_probabilities = np.empty((K, D))

        for k in range(K):
            for d in range(D):
                expected_emission_probabilities[k, d] = np.sum(sample_z_star[:, k] *
                                                               sample_dataset[:, d]) / u[k]

        # First, perform the test with the same dataset, and weights set to one
        mc_ones, ep_ones = _m_step(sample_dataset,
                                   np.ones(sample_dataset.shape, dtype=bool),
                                   sample_z_star,
                                   np.ones(N, dtype=int),
                                   np.empty(expected_emission_probabilities.shape))

        assert_array_almost_equal(expected_mixing_coefficients, mc_ones)
        assert_array_almost_equal(expected_emission_probabilities, ep_ones)

        # Use unique dataset & weights to compute values for test.

        mixing_coefficients, emission_probabilities = _m_step(unique_dataset,
                                                              np.ones(unique_dataset.shape, dtype=bool),
                                                              unique_z_star,
                                                              weights,
                                                              np.empty(expected_emission_probabilities.shape))

        assert_array_almost_equal(expected_mixing_coefficients, mixing_coefficients)
        assert_array_almost_equal(expected_emission_probabilities, emission_probabilities)

    def test_m_step_computes_correct_parameters_with_mask(self):
        sample_z_star = np.array([[0.24522862, 0.50447031, 0.25030106],
                                  [0.3264654, 0.67158596, 0.00194864],
                                  [0.24522862, 0.50447031, 0.25030106],
                                  [0.3264654, 0.67158596, 0.00194864],
                                  [0.55562318, 0.4286236, 0.01575322],
                                  [0.24522862, 0.50447031, 0.25030106]])

        unique_z_star = np.array([[0.24522862, 0.50447031, 0.25030106],
                                  [0.3264654, 0.67158596, 0.00194864],
                                  [0.55562318, 0.4286236, 0.01575322]])

        old_ps = np.array([[0.5, 0.5, 0.5, 0.5],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8]])

        sample_dataset = np.array([[True, True, False, False],  # 1
                                   [False, True, False, False],  # 2
                                   [True, True, False, False],  # 1
                                   [False, True, False, False],  # 2
                                   [False, False, False, False],  # 3
                                   [True, True, False, False]])  # 1

        unique_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   ])

        mask = np.array([[True, True, True, True],
                         [False, True, True, False],
                         [True, True, True, True],
                         [False, True, True, False],
                         [False, False, False, False],
                         [True, True, True, True]])

        unique_mask = np.array([[True, True, True, True],
                                [False, True, True, False],
                                [False, False, False, False]])

        # Flipping masked values should have no effect.
        sample_dataset2 = sample_dataset.copy()
        sample_dataset2[~mask] = ~sample_dataset2[~mask]
        unique_dataset2 = unique_dataset.copy()
        unique_dataset2[~unique_mask] = ~unique_dataset2[~unique_mask]

        weights = np.array([3, 2, 1], dtype=int)

        # -- Compute correct parameters from the full dataset

        N, D = sample_dataset.shape
        __, K = sample_z_star.shape

        u = np.sum(sample_z_star, axis=0)

        expected_mixing_coefficients = u / N

        expected_emission_probabilities = np.empty((K, D))

        for k in range(K):
            for d in range(D):
                data = np.array(sample_dataset[:, d], dtype=float)

                data[~mask[:, d]] = old_ps[k, d]

                expected_emission_probabilities[k, d] = np.sum(sample_z_star[:, k] * data) / u[k]

        # First, perform the test with the same dataset, and weights set to one
        mc_ones, ep_ones = _m_step(sample_dataset,
                                   mask,
                                   sample_z_star,
                                   np.ones(N, dtype=int),
                                   old_ps)
        assert_array_almost_equal(expected_mixing_coefficients, mc_ones)
        assert_array_almost_equal(expected_emission_probabilities, ep_ones)

        mc_ones2, ep_ones2 = _m_step(sample_dataset2,
                                     mask,
                                     sample_z_star,
                                     np.ones(N, dtype=int),
                                     old_ps)
        assert_array_almost_equal(expected_mixing_coefficients, mc_ones2)
        assert_array_almost_equal(expected_emission_probabilities, ep_ones2)

        # Use unique dataset & weights to compute values for test.

        mcs, eps = _m_step(unique_dataset,
                           unique_mask,
                           unique_z_star,
                           weights,
                           old_ps)

        assert_array_almost_equal(expected_mixing_coefficients, mcs)
        assert_array_almost_equal(expected_emission_probabilities, eps)

        mcs2, eps2 = _m_step(unique_dataset2,
                             unique_mask,
                             unique_z_star,
                             weights,
                             old_ps)

        assert_array_almost_equal(expected_mixing_coefficients, mcs2)
        assert_array_almost_equal(expected_emission_probabilities, eps2)


class TestImputation(unittest.TestCase):

    def test_missing_values_imputed_correctly(self):

        pis = np.array([0.5, 0.25, 0.25])
        ps = np.array([[0.5, 0.3],
                       [0.3, 0.2],
                       [0.9, 0.1]])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        # Unobserved values should not impact anything, so they can be flipped
        data2 = np.array([[True, True],
                         [False, False],
                         [False, True],
                         [True, False]])

        mask = np.array([[True, True],
                         [False, True],
                         [True, False],
                         [False, False]])

        S = probability_z_o_given_theta_c(data, mask, ps, pis)
        expected_answer = np.array([[True, True],
                                   [np.sum(ps[:, 0] * S[1, :]) / np.sum(S[1, :]), False],
                                   [False, np.sum(ps[:, 1] * S[2, :]) / np.sum(S[2, :])],
                                   [np.sum((ps[:, 0] * S[3, :])) / np.sum(S[3, :]),
                                    np.sum((ps[:, 1] * S[3, :])) / np.sum(S[3, :])]])

        actual_answer = impute_missing_data_c(data, mask, ps, pis)
        actual_answer2 = impute_missing_data_c(data2, mask, ps, pis)

        assert_array_almost_equal(expected_answer, actual_answer)
        assert_array_almost_equal(expected_answer, actual_answer2)

    def test_imputation_from_pandas(self):

        pis = np.array([0.5, 0.25, 0.25])
        ps = np.array([[0.5, 0.3],
                       [0.3, 0.2],
                       [0.9, 0.1]])

        data = pd.DataFrame(np.array([[True, True],
                                      [None, False],
                                      [False, None],
                                      [None, None]]),
                            index=['a', 'b', 'c', 'd'], columns=['aa', 'bb'])


        arr, mask = BernoulliMixture._as_decoupled_array(data)

        S = probability_z_o_given_theta_c(arr, mask, ps, pis)

        expected_answer = pd.DataFrame(np.array([[True, True],
                                                 [np.sum(ps[:, 0] * S[1, :]) / np.sum(S[1, :]),
                                                  False],
                                                 [False,
                                                  np.sum(ps[:, 1] * S[2, :]) / np.sum(S[2, :])],
                                                 [np.sum((ps[:, 0] * S[3, :])) / np.sum(S[3, :]),
                                                  np.sum((ps[:, 1] * S[3, :])) / np.sum(S[3, :])]]),
                                       index=data.index, columns=data.columns)

        mixture = BernoulliMixture(pis, ps)
        actual_answer = mixture.impute_missing_values(data)

        self.assertIsInstance(actual_answer, pd.DataFrame)
        assert_array_almost_equal(actual_answer.values, expected_answer.values)
        self.assertTrue(expected_answer.index.equals(actual_answer.index))
        self.assertTrue(expected_answer.columns.equals(actual_answer.columns))

