from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from bernoullimix.random_initialisation import _adjust_probabilities


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
