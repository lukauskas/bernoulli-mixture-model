import unittest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from bernoullimix._bernoulli import probability_z_o_given_theta_c, bernoulli_prob_for_observations, bernoulli_prob_for_observations_with_mask


class TestBernoulliEmissionProbabilities(unittest.TestCase):

    def test_all_observed_no_mask(self):
        ps = np.array([0.3, 0.45])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        expected_answer = np.array([0.3 * 0.45, 0.3 * 0.55, 0.7 * 0.55, 0.7 * 0.45])

        actual_answer = bernoulli_prob_for_observations(ps, data)

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

    def test_observed_joint_computation_all_observed(self):

        pis = np.array([0.5, 0.25, 0.25])
        ps = np.array([[0.5, 0.3],
                       [0.3, 0.2],
                       [0.9, 0.1]])

        data = np.array([[True, True],
                         [True, False],
                         [False, False],
                         [False, True]])

        observed_mask = np.ones((4, 2), dtype=bool)

        expected_answer = np.array([[pis[0] * 0.5 * 0.3, pis[1] * 0.3 * 0.2, pis[2] * 0.9 * 0.1],
                                    [pis[0] * 0.5 * 0.7, pis[1] * 0.3 * 0.8, pis[2] * 0.9 * 0.9],
                                    [pis[0] * 0.5 * 0.7, pis[1] * 0.7 * 0.8, pis[2] * 0.1 * 0.9],
                                    [pis[0] * 0.5 * 0.3, pis[1] * 0.7 * 0.2, pis[2] * 0.1 * 0.1]])

        actual_answer = probability_z_o_given_theta_c(data, ps, pis)

        assert_array_almost_equal(expected_answer, actual_answer)

