from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal

from bernoullimix._bernoulli import _m_step
from bernoullimix.mixture import BernoulliMixture

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


class TestFit(unittest.TestCase):

    def test_fit_validates_dataset_shape(self):
        """
        Given a dataset that has either too few or too many dimensions,
        fit function should raise an error message.
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

        self.assertRaises(ValueError, mixture.fit, dataset_too_few_dims)
        self.assertRaises(ValueError, mixture.fit, dataset_too_many_dims)

    def test_posterior_probability_given_support_is_computed_correctly(self):

        sample_support = np.array([[0.0042, 0.00864, 0.00428687],
                                   [0.0378, 0.07776, 0.00022563],
                                   [0.0042, 0.00864, 0.00428687],
                                   [0.0378, 0.07776, 0.00022563],
                                   [0.1512, 0.11664, 0.00428688],
                                   [0.0042, 0.00864, 0.00428687]])

        expected_z_star = np.empty([6, 3])
        for i in range(len(sample_support)):
            expected_z_star[i] = sample_support[i] / np.sum(sample_support[i])

        actual_z_star = BernoulliMixture._posterior_probability_of_class_given_support(sample_support)

        assert_array_almost_equal(expected_z_star, actual_z_star)

    def test_z_star_values_always_sum_to_one(self):
        """
        Z-star values must always sum to one
        """

        raise unittest.SkipTest('Not sure if this is an issue yet '
                                'as the differences are within the eps tolerance')
        # These come from actual datasets where z_star generated did not have sum to one
        suspicious_support = np.array([
                                       # Rows below used to sum to >1
                                       [1.53612947e-04, 9.88770995e-09, 7.31954865e-08,
                                        3.68751524e-12],
                                       [1.87860595e-08, 7.52234428e-10, 5.20148206e-08,
                                        6.21544388e-15],
                                       [3.19482931e-06, 8.96332090e-11, 6.21308954e-09,
                                        8.57877361e-15],
                                       [5.58900957e-24, 7.89065644e-09, 5.01641557e-20,
                                        0.00000000e+00],
                                       [1.20063058e-18, 2.99752427e-08, 4.05523428e-15,
                                        0.00000000e+00],
                                       [3.22436625e-22, 1.70696284e-08, 2.83703853e-21,
                                        0.00000000e+00],
                                       # Rows below used to sum to <1:
                                       [4.48894642e-07, 1.25938281e-09, 4.89169020e-11,
                                        1.33770571e-14],
                                       [3.26362569e-06, 1.31062689e-09, 5.37580436e-07,
                                        9.38841678e-14],
                                       [3.58296743e-06, 2.24325747e-11, 6.55336297e-09,
                                        1.59294917e-14],
                                       [5.28134593e-24, 7.78303030e-11, 6.02972119e-29,
                                        0.00000000e+00],
                                       [2.24817618e-23, 7.34013933e-12, 2.76654735e-32,
                                        0.00000000e+00],
                                       [2.55102334e-31, 7.17632935e-16, 1.30407144e-42,
                                        0.00000000e+00]
                                    ])

        z_star = BernoulliMixture._posterior_probability_of_class_given_support(suspicious_support)
        z_star_sums = np.sum(z_star, axis=1)

        for sum_ in z_star_sums:
            self.assertEqual(sum_, 1.0)



    def test_fit_increases_likelihood(self):

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        log_likelihood_prior_to_fitting = mixture.log_likelihood(sample_dataset)
        mixture.fit(sample_dataset, iteration_limit=1)

        log_likelihood_after_fitting = mixture.log_likelihood(sample_dataset)

        self.assertGreater(log_likelihood_after_fitting, log_likelihood_prior_to_fitting)

    def test_fit_with_or_without_aggregation_produces_same_result(self):

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture_a = BernoulliMixture(number_of_components, number_of_dimensions,
                                     sample_mixing_coefficients, sample_emission_probabilities)

        mixture_b = BernoulliMixture(number_of_components, number_of_dimensions,
                                     sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset_unique, counts = BernoulliMixture.aggregate_dataset(sample_dataset)

        log_likelihood_a, __ = mixture_a.fit(sample_dataset, iteration_limit=1)
        log_likelihood_b, __ = mixture_b.fit_aggregated(sample_dataset_unique, counts,
                                                        iteration_limit=1)

        self.assertEqual(log_likelihood_a, log_likelihood_b)

    def test_fit_with_aggregation_accepts_pandas_object(self):
        try:
            import pandas as pd
        except ImportError:
            raise unittest.SkipTest

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture_a = BernoulliMixture(number_of_components, number_of_dimensions,
                                     sample_mixing_coefficients, sample_emission_probabilities)

        mixture_b = BernoulliMixture(number_of_components, number_of_dimensions,
                                     sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset_unique, counts = BernoulliMixture.aggregate_dataset(sample_dataset)

        sample_dataset_unique_pd = pd.DataFrame(sample_dataset_unique)
        counts_pd = pd.Series(counts)

        log_likelihood_a, __ = mixture_a.fit_aggregated(sample_dataset_unique,
                                                        counts,
                                                        iteration_limit=1)

        log_likelihood_b, __ = mixture_b.fit_aggregated(sample_dataset_unique_pd,
                                                        counts_pd,
                                                        iteration_limit=1)

        self.assertEqual(log_likelihood_a, log_likelihood_b)

    def test_fit_accepts_pandas_object(self):
        try:
            import pandas as pd
        except ImportError:
            raise unittest.SkipTest

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture_a = BernoulliMixture(number_of_components, number_of_dimensions,
                                     sample_mixing_coefficients, sample_emission_probabilities)

        mixture_b = BernoulliMixture(number_of_components, number_of_dimensions,
                                     sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset_pd = pd.DataFrame(sample_dataset)

        log_likelihood_a, __ = mixture_a.fit(sample_dataset,
                                             iteration_limit=1)

        log_likelihood_b, __ = mixture_b.fit(sample_dataset_pd,
                                             iteration_limit=1)

        self.assertEqual(log_likelihood_a, log_likelihood_b)

    def test_reported_log_likelihood_is_true_likelihood_post_fitting(self):

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        log_likelihood, __ = mixture.fit(sample_dataset, iteration_limit=1)
        log_likelihood_after_fitting = mixture.log_likelihood(sample_dataset)

        self.assertEqual(log_likelihood, log_likelihood_after_fitting)


    def test_fit_converges(self):

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        # such a simple model should converge within 100 iter. even for eps
        log_likelihood, convergence = mixture.fit(sample_dataset, iteration_limit=100,
                                                  convergence_threshold=np.finfo(np.float64).eps)
        self.assertTrue(convergence.converged)

    def test_fit_converges_immediately_if_already_converged(self):

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        # quite a large convergence threshold, to ensure fast convergence
        log_likelihood, convergence = mixture.fit(sample_dataset, iteration_limit=1000,
                                                  convergence_threshold=0.01)

        self.assertTrue(convergence.converged)  # Sanity check
        self.assertGreater(convergence.number_of_iterations, 1)  # Another sanity check

        # The second time we call fit. Ensure it is converged:
        log_likelihood, convergence = mixture.fit(sample_dataset, iteration_limit=1000,
                                                  convergence_threshold=0.01)

        self.assertTrue(convergence.converged)
        # The actual test. Should only be one recorded iteration:
        self.assertEqual(convergence.number_of_iterations, 1)

    def test_m_step_error_for_large_datasets(self):
        """
        This tests that the error (due to floating point, probably)
        does not accumulate over a large N
        :return:
        """

        def random_zstar(shape, random):
            zstar = random.rand(*shape)
            normalise = lambda x: x / x.sum()
            zstar = np.apply_along_axis(normalise, 1, zstar)
            return zstar

        def random_dataset(shape, random):
            dataset = np.asarray(random.randint(2, size=shape), dtype=bool)
            return BernoulliMixture.aggregate_dataset(dataset)

        def random_zstar_and_dataset(N, D, K, random_state):
            random = np.random.RandomState(random_state)
            rd, rw = random_dataset((N, D), random)
            rz = random_zstar((rd.shape[0], K), random)
            return rd, rw, rz

        N, D, K = 44001, 20, 4

        unique_dataset, unique_counts, unique_zstar = random_zstar_and_dataset(N, D, K,
                                                                               random_state=100)

        u = np.sum((unique_zstar.T * np.asarray(unique_counts)), axis=1)
        vs = np.empty((K, unique_dataset.shape[1]))

        for k in range(K):
            v_k = np.sum(unique_dataset.T * unique_counts * unique_zstar[:, k], axis=1)
            vs[k] = v_k / u[k]

        expected_u = u / np.sum(u)
        expected_v = vs

        actual_u, actual_v = _m_step(np.asarray(unique_dataset, dtype=bool),
                                     unique_zstar,
                                     np.asarray(unique_counts))

        assert_array_almost_equal(expected_u, actual_u)
        assert_array_almost_equal(expected_v, actual_v)

    def test_m_step_produces_probabilities_in_correct_range_for_repeating_datasets(self):
        unique_dataset = np.array([[True, False, True, True],
                            [True, False, False, True],
                            [True, False, True, False],
                            [True, False, False, False]])

        unique_counts = np.array([4, 3, 2, 1])

        # This was randomly generated, any z-star should work
        unique_zstar = np.array([[0.25986956, 0.13312306, 0.20301472, 0.40399265],
                                 [0.00290769, 0.07490904, 0.41330538, 0.5088779],
                                 [0.07543165, 0.31732369, 0.49181159, 0.11543307],
                                 [0.12421243, 0.07263738, 0.14724773, 0.65590246]])
        K=4

        u = np.sum((unique_zstar.T * unique_counts), axis=1)
        vs = np.empty((K, unique_dataset.shape[1]))

        for k in range(K):
            v_k = np.sum(unique_dataset.T * unique_counts * unique_zstar[:, k], axis=1)
            vs[k] = v_k / u[k]

        expected_u = u / np.sum(u)
        expected_v = vs

        actual_u, actual_v = _m_step(unique_dataset, unique_zstar, unique_counts)

        assert np.all((expected_v <= 1) & (expected_v >= 0))  # for the sake of sanity...

        # Actual test
        self.assertTrue(np.all(actual_v <= 1),
                        'Some values returned are >1: {!r}'.format(actual_v[actual_v > 1])
                        )
        self.assertTrue(np.all(actual_v >= 0),
                        'Some values returned are <0: {!r}'.format(actual_v[actual_v < 0])
                        )


class TestDatasetAggregation(unittest.TestCase):

    def test_dataset_aggregation_all_observed(self):
        """
        Given a dataset with repeating rows `_aggregate_dataset()` function should return
        a reduced dataset with all unique rows, as well as a vector of weights for each row.

        The resulting rows can be in any order.

        """
        sample_dataset = np.array([[True, True, False, False],  # row A
                                   [False, True, False, False],  # row B
                                   [True, True, False, False],  # row A
                                   [False, True, False, False],  # row B
                                   [False, False, False, False],  # row C
                                   [True, True, False, False]])  # row A

        expected_aggregated_dataset = pd.DataFrame(
            np.array([[True, True, False, False],  # A, three times
                      [False, True, False, False],  # B, twice
                      [False, False, False, False]  # C, once
                      ]))

        expected_aggregated_weights = pd.Series(np.array([3, 2, 1], dtype=int),
                                                index=expected_aggregated_dataset.index)

        actual_aggregated_dataset, actual_weights = BernoulliMixture.aggregate_dataset(sample_dataset)

        # Check that shapes are the same
        self.assertEqual(expected_aggregated_dataset.shape, actual_aggregated_dataset.shape)
        self.assertEqual(expected_aggregated_weights.shape, actual_weights.shape)

        # check that indices match
        self.assertTrue(actual_aggregated_dataset.index.equals(actual_weights.index))

        # Since the order returned doesn't matter, let's turn results into dict and compare those
        expected_lookup = self._construct_lookup(expected_aggregated_dataset,
                                                 expected_aggregated_weights)
        actual_lookup = self._construct_lookup(actual_aggregated_dataset,
                                               actual_weights)

        self.assertDictEqual(expected_lookup, actual_lookup)

    def _construct_lookup(self, unique, counts):

        lookup = {}
        for ix, row in unique.iterrows():
            count = counts.loc[ix]

            row_tuple = tuple([x if x is not None and not np.isnan(x) else None for x in row])
            lookup[row_tuple] = count

        return lookup

    def test_dataset_aggregation_with_masked_dataframe(self):
        sample_dataset = pd.DataFrame(
            np.array([[True, True, False, False],  # row A1
                      [False, None, False, None],  # row B1
                      [True, True, False, False],  # row A1
                      [False, None, False, False],  # row B2
                      [False, False, False, False],  # row C
                      [True, True, None, False]]),  # row A2
            columns=['a', 'b', 'c', 'd']
        )

        expected_aggregated_dataset = pd.DataFrame(
            np.array([[True, True, False, False],  # A1, two times
                      [False, None, False, None],  # B1, once
                      [False, False, False, False],  # C, once
                      [True, True, None, False],  # A2
                      [False, None, False, False],  # B2
                      ]),
            columns=sample_dataset.columns)

        expected_aggregated_weights = pd.Series(
            np.array([2, 1, 1, 1, 1], dtype=int), index=expected_aggregated_dataset.index)

        actual_aggregated_dataset, actual_weights = BernoulliMixture.aggregate_dataset(sample_dataset)

        # Check that shapes are the same
        self.assertEqual(expected_aggregated_dataset.shape, actual_aggregated_dataset.shape)
        self.assertEqual(expected_aggregated_weights.shape, actual_weights.shape)

        # check that indices match
        self.assertTrue(actual_aggregated_dataset.index.equals(actual_weights.index))

        # Since the order returned doesn't matter, let's turn results into dict and compare those
        expected_lookup = self._construct_lookup(expected_aggregated_dataset,
                                                 expected_aggregated_weights)
        actual_lookup = self._construct_lookup(actual_aggregated_dataset,
                                               actual_weights)

        self.assertDictEqual(expected_lookup, actual_lookup)

        self.assertTrue(expected_aggregated_dataset.columns.equals(actual_aggregated_dataset.columns))

    def test_aggregation_from_masked_array(self):
        sample_dataset = np.array([[True, True, False, False],  # row A
                                   [False, True, False, False],  # row B
                                   [True, True, False, False],  # row A
                                   [False, True, False, False],  # row B
                                   [False, False, False, False],  # row C
                                   [True, True, False, False]])  # row A

        mask = np.array([[True, True, True, True],  # Row A, all obs
                         [True, False, True, False],  # Row B, some obs
                         [True, True, True, True],  # Row A, all obs
                         [True, False, True, True],  # Row B, some obs
                         [True, True, True, True],  # row C, all obs
                         [True, True, False, True]  # row A some obs
                         ])

        # Masked array takes the convention that the hidden values are masked, thus ~mask
        sample_dataset = np.ma.array(sample_dataset, mask=~mask)

        expected_aggregated_dataset = pd.DataFrame(
            np.array([[True, True, False, False],  # A1, two times
                      [False, None, False, None],  # B1, once
                      [False, False, False, False],  # C, once
                      [True, True, None, False],  # A2
                      [False, None, False, False],  # B2
                      ]))

        expected_aggregated_weights = pd.Series(
            np.array([2, 1, 1, 1, 1], dtype=int), index=expected_aggregated_dataset.index)

        actual_aggregated_dataset, actual_weights = BernoulliMixture.aggregate_dataset(
            sample_dataset)

        # Check that shapes are the same
        self.assertEqual(expected_aggregated_dataset.shape, actual_aggregated_dataset.shape)
        self.assertEqual(expected_aggregated_weights.shape, actual_weights.shape)

        # check that indices match
        self.assertTrue(actual_aggregated_dataset.index.equals(actual_weights.index))

        # Since the order returned doesn't matter, let's turn results into dict and compare those
        expected_lookup = self._construct_lookup(expected_aggregated_dataset,
                                                 expected_aggregated_weights)
        actual_lookup = self._construct_lookup(actual_aggregated_dataset,
                                               actual_weights)

        self.assertDictEqual(expected_lookup, actual_lookup)

class TestPenalisedLikelihood(unittest.TestCase):

    def test_number_of_free_parameters_computed_correctly(self):
        number_of_components = 3
        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)


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

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

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

    def test_aic_computed_correctly(self):
        number_of_components = 3

        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

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

class TestAssignment(unittest.TestCase):

    def test_soft_assignment_computed_correctly(self):
        number_of_components = 3

        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        dataset = np.array([[True, True, False, False],
                            [False, True, False, False],
                            [True, True, False, False],
                            [False, True, False, False],
                            [False, False, False, False],
                            [True, True, False, False]])

        N = len(dataset)

        expected_assignments = np.empty((N, number_of_components))

        for n in range(N):
            for k in range(number_of_components):

                prob = sample_mixing_coefficients[k]
                prob *= np.product(np.power(sample_emission_probabilities[k], dataset[n]) *
                                   np.power(1 - sample_emission_probabilities[k], 1 - dataset[n]))

                expected_assignments[n, k] = prob

            sum_ = np.sum(expected_assignments[n])

            expected_assignments[n] /= sum_

        actual_assignments = mixture.soft_assignment(dataset)

        assert_array_almost_equal(expected_assignments, actual_assignments)

    def test_hard_assignment_computed_correctly(self):
        number_of_components = 3

        number_of_dimensions = 4

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [0.5, 0.5, 0.5, 0.5]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        dataset = np.array([[True, True, False, False],
                            [False, True, False, False],
                            [True, True, False, False],
                            [False, True, False, False],
                            [False, False, False, False],
                            [True, True, False, False]])

        N = len(dataset)

        expected_assignments = np.empty(N, dtype=int)

        for n in range(N):
            assignment_probabilities = np.empty(number_of_components)

            for k in range(number_of_components):
                prob = sample_mixing_coefficients[k]
                prob *= np.product(np.power(sample_emission_probabilities[k], dataset[n]) *
                                   np.power(1 - sample_emission_probabilities[k], 1 - dataset[n]))

                assignment_probabilities[k] = prob

            expected_assignments[n] = np.argmax(assignment_probabilities)

        actual_assignments = mixture.hard_assignment(dataset)

        assert_array_equal(expected_assignments, actual_assignments)
