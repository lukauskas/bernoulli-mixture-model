from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import pandas as pd

from bernoullimix import MultiDatasetMixtureModel


class TestCommonOperations(unittest.TestCase):

    def test_equality_returns_true_if_all_equal(self):

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

        model_a = MultiDatasetMixtureModel(mu, pi, p)
        model_b = MultiDatasetMixtureModel(mu, pi, p)

        self.assertEqual(model_a, model_b)

    def test_equality_false_if_mus_different(self):
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
        mu_b = pd.Series([0.25, 0.5, 0.25], index=['dataset-a', 'dataset-b', 'dataset-c'])

        model_a = MultiDatasetMixtureModel(mu, pi, p)
        model_b = MultiDatasetMixtureModel(mu_b, pi, p)

        self.assertNotEqual(model_a, model_b)

    def test_equality_false_if_pis_different(self):
        pi = pd.DataFrame([[0.6, 0.4],
                           [0.2, 0.8],
                           [0.5, 0.5]],
                          index=['dataset-a', 'dataset-b', 'dataset-c'],
                          columns=['K0', 'K1'])
        pi_b = pd.DataFrame([[0.4, 0.6],
                           [0.2, 0.8],
                           [0.5, 0.5]],
                          index=['dataset-a', 'dataset-b', 'dataset-c'],
                          columns=['K0', 'K1'])

        p = pd.DataFrame([[0.1, 0.2, 0.3],
                          [0.9, 0.8, 0.7]],
                         index=['K0', 'K1'],
                         columns=['X1', 'X2', 'X3'])

        mu = pd.Series([0.5, 0.25, 0.25], index=['dataset-a', 'dataset-b', 'dataset-c'])

        model_a = MultiDatasetMixtureModel(mu, pi, p)
        model_b = MultiDatasetMixtureModel(mu, pi_b, p)

        self.assertNotEqual(model_a, model_b)

    def test_equality_false_if_ps_different(self):
        pi = pd.DataFrame([[0.6, 0.4],
                           [0.2, 0.8],
                           [0.5, 0.5]],
                          index=['dataset-a', 'dataset-b', 'dataset-c'],
                          columns=['K0', 'K1'])

        p = pd.DataFrame([[0.1, 0.2, 0.3],
                          [0.9, 0.8, 0.7]],
                         index=['K0', 'K1'],
                         columns=['X1', 'X2', 'X3'])

        p_b = pd.DataFrame([[0.2, 0.1, 0.3],
                            [0.9, 0.8, 0.7]],
                            index=['K0', 'K1'],
                            columns=['X1', 'X2', 'X3'])

        mu = pd.Series([0.5, 0.25, 0.25], index=['dataset-a', 'dataset-b', 'dataset-c'])

        model_a = MultiDatasetMixtureModel(mu, pi, p)
        model_b = MultiDatasetMixtureModel(mu, pi, p_b)

        self.assertNotEqual(model_a, model_b)
