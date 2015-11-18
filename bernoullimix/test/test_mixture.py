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
                                                  [0.95, 0.05, 0.05, 0.05]])

        mixture = BernoulliMixture(number_of_components, number_of_dimensions,
                                   sample_mixing_coefficients, sample_emission_probabilities)

        sample_dataset = np.array([[True, True, False, False],
                                   [False, True, False, False],
                                   [True, True, False, False],
                                   [False, True, False, False],
                                   [False, False, False, False],
                                   [True, True, False, False]])

        unique_dataset, weights = BernoulliMixture._aggregate_dataset(sample_dataset)

        # Compute support on whole dataset for the test
        # even though code would compute it for unique_dataset only
        support = mixture._observation_emission_support(sample_dataset)
        expected_answer = np.sum(np.log(np.sum(support, axis=1)))

        actual_answer = mixture.log_likelihood(sample_dataset)

        unique_support = mixture._observation_emission_support(unique_dataset)
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

    def test_z_step_computes_correct_parameters(self):
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
        mc_ones, ep_ones = BernoulliMixture._m_step(sample_z_star,
                                                    sample_dataset,
                                                    np.ones(N, dtype=int))

        assert_array_almost_equal(expected_mixing_coefficients, mc_ones)
        assert_array_almost_equal(expected_emission_probabilities, ep_ones)

        # Use unique dataset & weights to compute values for test.

        mixing_coefficients, emission_probabilities = BernoulliMixture._m_step(unique_z_star,
                                                                               unique_dataset,
                                                                               weights)

        assert_array_almost_equal(expected_mixing_coefficients, mixing_coefficients)
        assert_array_almost_equal(expected_emission_probabilities, emission_probabilities)


    def test_dataset_aggregation(self):
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

        expected_aggregated_dataset = np.array([[True, True, False, False],   # A, three times
                                                [False, True, False, False],  # B, twice
                                                [False, False, False, False]  # C, once
                                                ])

        expected_aggregated_weights = np.array([3, 2, 1], dtype=int)

        actual_aggregated_dataset, actual_weights = BernoulliMixture._aggregate_dataset(sample_dataset)

        # Check that shapes are the same
        self.assertEqual(expected_aggregated_dataset.shape, actual_aggregated_dataset.shape)
        self.assertEqual(expected_aggregated_weights.shape, actual_weights.shape)

        # Since the order returned doesn't matter, let's turn results into dict and compare those
        expected_lookup = dict(zip(map(tuple, expected_aggregated_dataset),
                                   expected_aggregated_weights))
        actual_lookup = dict(zip(map(tuple, actual_aggregated_dataset),
                                 actual_weights))
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
        actual_bic = mixture.BIC(dataset)

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
        actual_aic = mixture.AIC(dataset)

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
