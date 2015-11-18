from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from bernoullimix.random_initialisation import _adjust_probabilities, _expected_domain, \
    _random_numbers_within_domain


class TestProbabilityAdjustment(unittest.TestCase):

    def test_probabilities_are_adjusted_correctly_given_domain(self):

        domain = (-5, 5)
        epsilon = 0.05

        test_array = np.array([-5.0, 5.0, 0.0, 4.0])

        expected_result = np.array([epsilon,
                                    1-epsilon,
                                    0.5, ((4.0 - (-5))/10) * (1-2*epsilon) + epsilon])

        actual_result = _adjust_probabilities(test_array, epsilon, domain=domain)
        assert_array_almost_equal(expected_result, actual_result)

    def test_probabilities_are_adjusted_correctly_given_domain_2d(self):
        domain = (-5, 5)
        epsilon = 0.05

        test_array = np.array([[-5.0, 5.0],
                               [0.0, 4.0]])

        expected_result = np.array([[epsilon, 1 - epsilon],
                                    [0.5, ((4.0 - (-5)) / 10) * (1 - 2 * epsilon) + epsilon]])

        actual_result = _adjust_probabilities(test_array, epsilon, domain=domain)
        assert_array_almost_equal(expected_result, actual_result)

    def test_probabilities_out_of_domain_raise_value_error(self):
        epsilon = 0.05

        test_array = np.array([-5.0, 5.0, 0.0, 4.0])

        # Low side
        self.assertRaises(ValueError, _adjust_probabilities,
                          test_array, epsilon, domain=(-4.0, 5.0))

        # High side
        self.assertRaises(ValueError, _adjust_probabilities,
                          test_array, epsilon, domain=(-5.0, 4.0))

    def test_ill_specified_domain_raises_value_error(self):
        epsilon = 0.05

        test_array = np.array([-5.0, 5.0, 0.0, 4.0])

        # domain[0] > domain[1]
        self.assertRaises(ValueError, _adjust_probabilities,
                          test_array, epsilon, domain=(5.0, -5.0))

        # domain[0] == domain[1]
        self.assertRaises(ValueError, _adjust_probabilities,
                          test_array, epsilon, domain=(5.0, 5.0))

class TestDomainCalculation(unittest.TestCase):

    def test_domain_is_computed_correctly_when_alpha_is_zero_or_one(self):

        domain_a = (0, 1)
        domain_b = (-5, 5)

        expected_alpha_zero = domain_b
        expected_alpha_one = domain_a

        actual_alpha_zero = _expected_domain(domain_a, domain_b, alpha=0)
        actual_alpha_one = _expected_domain(domain_a, domain_b, alpha=1)

        self.assertEqual(expected_alpha_one, actual_alpha_one)
        self.assertEqual(expected_alpha_zero, actual_alpha_zero)

    def test_domain_computation_fails_for_bad_alpha(self):
        domain_a = (0, 1)
        domain_b = (-5, 5)

        # Negative alpha
        self.assertRaises(ValueError, _expected_domain, domain_a, domain_b,
                          alpha=-1)
        # Alpha greater than One
        self.assertRaises(ValueError,
                          _expected_domain, domain_a, domain_b,
                          alpha=1.1)

    def test_domain_stays_the_same_when_both_domains_equal(self):
        domain_a = domain_b = (0, 1)

        expected = domain_a

        # Regardless of alpha:
        actual_0_1 = _expected_domain(domain_a, domain_b, alpha=0.1)
        actual_0_3 = _expected_domain(domain_a, domain_b, alpha=0.3)
        actual_0_5 = _expected_domain(domain_a, domain_b, alpha=0.5)
        actual_0_7 = _expected_domain(domain_a, domain_b, alpha=0.7)

        self.assertEqual(expected, actual_0_1)
        self.assertEqual(expected, actual_0_3)
        self.assertEqual(expected, actual_0_5)
        self.assertEqual(expected, actual_0_7)

    def test_domain_computed_correctly_for_different_domains(self):
        domain_a = (-1, 1)
        domain_b = (0, 1)

        alpha = 0.75

        expected = (domain_a[0] * alpha + domain_b[0] * (1-alpha),
                    domain_a[1] * alpha + domain_b[1] * (1 - alpha))

        actual = _expected_domain(domain_a, domain_b, alpha)

        self.assertEqual(expected, actual)

class TestRandomNumberGeneration(unittest.TestCase):

    def _isin_domain_func(self, domain):
        return np.vectorize(lambda x: domain[0] <= x <= domain[1])

    def test_random_number_generator_produces_numbers_within_correct_domain(self):

        domain_a = (17.0, 130.0)
        domain_b = (-5.0, -4)

        random = np.random.RandomState(12345)

        numbers_a = _random_numbers_within_domain(random, domain_a, (10,))
        numbers_b = _random_numbers_within_domain(random, domain_b, (10,))

        isin_domain_a = self._isin_domain_func(domain_a)
        isin_domain_b = self._isin_domain_func(domain_b)

        all_in_a = np.all(isin_domain_a(numbers_a))
        all_in_b = np.all(isin_domain_b(numbers_b))

        self.assertTrue(all_in_a, 'Some numbers not in the domain {}:\n{!r}'.format(domain_a,
                                                                                    numbers_a))

        self.assertTrue(all_in_b, 'Some numbers not in the domain {}:\n{!r}'.format(domain_b,
                                                                                    numbers_b))

    def test_random_number_generator_samples_numbers_within_whole_domain(self):

        linspace = np.linspace(-1, 1, 10)
        complete_domain = linspace[0], linspace[-1]

        subdomains = zip(linspace, linspace[1:])

        random = np.random.RandomState(12345)

        numbers_to_generate = 10000

        numbers = _random_numbers_within_domain(random, complete_domain, (numbers_to_generate, ))

        for subdomain in subdomains:
            isin_subdomain = self._isin_domain_func(subdomain)

            some_in_subdomain = np.any(isin_subdomain(numbers))

            self.assertTrue(some_in_subdomain,
                            'Out of {:,} numbers generated for {} '
                            'none in domain {}'.format(numbers_to_generate,
                                                       complete_domain,
                                                       subdomain))

