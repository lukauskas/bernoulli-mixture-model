from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from bernoullimix.mixture import BernoulliMixture, MixtureModel
import pandas as pd

from pandas.util.testing import assert_frame_equal


class TestMixtureModelInitialisation(unittest.TestCase):
    def test_initialisation_with_array(self):
        number_of_components = 3
        number_of_dimensions = 4
        number_of_datasets = 1

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        mixture = MixtureModel(sample_mixing_coefficients, sample_emission_probabilities)

        self.assertEqual(number_of_components, mixture.n_components)
        self.assertEqual(number_of_dimensions, mixture.n_dimensions)
        self.assertEqual(number_of_datasets, mixture.n_datasets)

        self.assertIsInstance(mixture.mixing_coefficients, pd.DataFrame)
        self.assertIsInstance(mixture.emission_probabilities, pd.DataFrame)

        assert_array_equal(np.asarray(mixture.emission_probabilities),
                           sample_emission_probabilities)
        assert_array_equal(np.asarray(mixture.mixing_coefficients.iloc[0]),
                           sample_mixing_coefficients)

        self.assertTrue(mixture.mixing_coefficients.columns.equals(range(3)))
        self.assertTrue(mixture.mixing_coefficients.index.equals(range(1)))
        self.assertTrue(mixture.emission_probabilities.index.equals(range(3)))

    def test_initialisation_with_series(self):

        number_of_components = 3
        number_of_dimensions = 4
        number_of_datasets = 1

        sample_mixing_coefficients = pd.Series(np.array([0.5, 0.4, 0.1]), index=['c_a', 'c_b', 'c_c'])
        sample_mixing_coefficients.index.name = 'Components'

        sample_emission_probabilities = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.4],
                                                               [0.1, 0.4, 0.1, 0.4],
                                                               [1.0, 0.0, 0.0, 0.0]]),
                                                     index=sample_mixing_coefficients.index,
                                                     columns=['a', 'b', 'c', 'd'])

        mixture = MixtureModel(sample_mixing_coefficients, sample_emission_probabilities)

        self.assertEqual(number_of_components, mixture.n_components)
        self.assertEqual(number_of_dimensions, mixture.n_dimensions)
        self.assertEqual(number_of_datasets, mixture.n_datasets)

        assert_frame_equal(sample_emission_probabilities, mixture.emission_probabilities)
        assert_array_equal(np.asarray(sample_mixing_coefficients),
                           np.asarray(mixture.mixing_coefficients.iloc[0]))

        self.assertTrue(mixture.mixing_coefficients.index.equals(range(1)))
        self.assertTrue(mixture.mixing_coefficients.columns.equals(sample_mixing_coefficients.index))

    def test_initialisation_with_df(self):

        number_of_components = 3
        number_of_dimensions = 4
        number_of_datasets = 2

        sample_mixing_coefficients = pd.DataFrame(
            np.array([[0.5, 0.4, 0.1], [0.3, 0.2, 0.5]]),
            index=['Dataset 1', 'Dataset 2'],
            columns=['c_a', 'c_b', 'c_c'])

        sample_emission_probabilities = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.4],
                                                               [0.1, 0.4, 0.1, 0.4],
                                                               [1.0, 0.0, 0.0, 0.0]]),
                                                     index=sample_mixing_coefficients.columns,
                                                     columns=['a', 'b', 'c', 'd'])

        mixture = MixtureModel(sample_mixing_coefficients, sample_emission_probabilities)

        self.assertEqual(number_of_components, mixture.n_components)
        self.assertEqual(number_of_dimensions, mixture.n_dimensions)
        self.assertEqual(number_of_datasets, mixture.n_datasets)

        assert_frame_equal(sample_emission_probabilities, mixture.emission_probabilities)
        assert_frame_equal(sample_mixing_coefficients, mixture.mixing_coefficients)

    def test_error_is_raised_if_dims_dont_match(self):

        sample_mixing_coefficients_series = pd.Series(np.array([0.3, 0.4, 0.1, 0.2]), index=['c_a',
                                                                                             'c_b',
                                                                                             'c_c',
                                                                                             'c_d'])

        sample_mixing_coefficients_df = pd.DataFrame(np.array([[0.15, 0.75, 0.05, 0.05],
                                                               [0.15, 0.75, 0.05, 0.05],
                                                               [0.15, 0.75, 0.05, 0.05]]))

        sample_emission_probabilities = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.4],
                                                               [0.1, 0.4, 0.1, 0.4],
                                                               [1.0, 0.0, 0.0, 0.0]]),
                                                     columns=['a', 'b', 'c', 'd'])

        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_series,
                          sample_emission_probabilities)

        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_df,
                          sample_emission_probabilities)

    def assert_error_is_raised_if_indices_dont_match(self):

        sample_mixing_coefficients_series = pd.Series(np.array([0.5, 0.4, 0.1]), index=['c_a',
                                                                                        'c_b',
                                                                                        'c_c'])

        sample_mixing_coefficients_df = pd.DataFrame(np.array([[0.15, 0.75, 0.1],
                                                               [0.15, 0.75, 0.1],
                                                               [0.15, 0.75, 0.1]]),
                                                     columns=['c_a', 'c_b', 'c_c'])

        sample_emission_probabilities = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.4],
                                                               [0.1, 0.4, 0.1, 0.4],
                                                               [1.0, 0.0, 0.0, 0.0]]),
                                                     columns=['a', 'b', 'c', 'd'],
                                                     index=['a', 'b', 'c'])

        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_series,
                          sample_emission_probabilities)

        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_df,
                          sample_emission_probabilities)

    def test_error_is_raised_if_mixing_components_dont_sum_to_one(self):

        sample_mixing_coefficients_series_a = pd.Series(np.array([0.5, 0.4, 0.05]))
        sample_mixing_coefficients_series_b = pd.Series(np.array([0.5, 0.4, 0.15]))

        sample_mixing_coefficients_df_a = pd.DataFrame(np.array([[0.15, 0.75, 0.1],
                                                                 [0.15, 0.75, 0.05],
                                                                 [0.15, 0.75, 0.1]]))

        sample_mixing_coefficients_df_b = pd.DataFrame(np.array([[0.15, 0.75, 0.1],
                                                                 [0.15, 0.75, 0.15],
                                                                 [0.15, 0.75, 0.1]]))

        sample_emission_probabilities = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.4],
                                                               [0.1, 0.4, 0.1, 0.4],
                                                               [1.0, 0.0, 0.0, 0.0]]))

        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_series_a,
                          sample_emission_probabilities)
        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_series_b,
                          sample_emission_probabilities)
        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_df_a,
                          sample_emission_probabilities)
        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_df_b,
                          sample_emission_probabilities)

    def test_error_is_raised_if_emission_probs_below_zero_or_above_one(self):

        sample_mixing_coefficients_series = pd.Series(np.array([0.5, 0.4, 0.1]))

        sample_emission_probabilities_a = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.4],
                                                                [0.1, 0.4, 1.2, 0.4],
                                                                [1.0, 0.0, 0.0, 0.0]]))

        sample_emission_probabilities_b = pd.DataFrame(np.array([[0.1, 0.2, 0.3, -0.1],
                                                                [0.1, 0.4, 0.1, 0.4],
                                                                [1.0, 0.0, 0.0, 0.0]]))

        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_series,
                          sample_emission_probabilities_a)
        self.assertRaises(ValueError, MixtureModel,
                          sample_mixing_coefficients_series,
                          sample_emission_probabilities_b)

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

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

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

        mixture = BernoulliMixture(sample_mixing_coefficients, sample_emission_probabilities)

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

        too_few_components = np.array([0.5, 0.5])
        too_many_components = np.array([0.25, 0.25, 0.25, 0.25])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture,
                          too_few_components, sample_emission_probabilities)
        self.assertRaises(ValueError, BernoulliMixture,
                          too_many_components, sample_emission_probabilities)

    def test_constant_initialisation_when_mixing_coeffiecients_do_not_sum_to_one(self):
        """
        Given wrong number of mixing components, initialiser should raise an error.
        """

        less_than_one = np.array([0.5, 0.4, 0.05])
        more_than_one = np.array([0.25, 0.25, 0.7])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture,
                          less_than_one, sample_emission_probabilities)
        self.assertRaises(ValueError, BernoulliMixture,
                          more_than_one, sample_emission_probabilities)

    def test_constant_initialisation_when_mixing_coeffiecients_not_between_0_and_1(self):
        """
        Given wrong number of mixing components, initialiser should raise an error.
        """

        less_than_zero = np.array([-0.5, 1, 0.5])
        more_than_one = np.array([1.5, -1.2, 0.7])
        sample_emission_probabilities = np.array([[0.1, 0.2, 0.3, 0.4],
                                                  [0.1, 0.4, 0.1, 0.4],
                                                  [1.0, 0.0, 0.0, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture,
                          less_than_zero, sample_emission_probabilities)
        self.assertRaises(ValueError, BernoulliMixture,
                          more_than_one, sample_emission_probabilities)

    def test_constant_initialisation_when_emission_probabilities_are_bounded_appropriately(self):
        """
        Given that emission probabilities are greater than one or lower than one, raise error.
        """

        sample_mixing_coefficients = np.array([0.5, 0.4, 0.1])
        more_than_one = np.array([[0.1, 0.2, 0.3, 0.4],
                                  [0.1, 0.4, 0.1, 0.4],
                                  [1.0, 0.0, 1.8, 0.0]])

        less_than_zero = np.array([[0.1, 0.1, 0.3, 0.4],
                                  [0.1, 0.4, 0.1, 0.4],
                                  [1.0, 0.0, -5, 0.0]])

        self.assertRaises(ValueError, BernoulliMixture, sample_mixing_coefficients, more_than_one)
        self.assertRaises(ValueError, BernoulliMixture, sample_mixing_coefficients, less_than_zero)
