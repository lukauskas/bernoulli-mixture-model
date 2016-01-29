from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Counter

import numpy as np
import pandas as pd

from bernoullimix._mixture import partial_support, p_update_unnormalised

_EPSILON = np.finfo(np.float).eps

DATASET_ID_COLUMN = 'dataset_id'
WEIGHT_COLUMN = 'weight'


class MultiDatasetMixtureModel(object):

    _dataset_priors = None
    _mixing_coefficients = None
    _emission_probabilities = None

    def _validate_init(self):
        if not self.mixing_coefficients.columns.equals(self.emission_probabilities.index):
            raise ValueError('The mixing coefficients index does not match emission probabilities '
                             'index {!r} != {!r}'.format(self.mixing_coefficients.columns,
                                                         self.emission_probabilities.index))

        mc_sums = self.mixing_coefficients.sum(axis='columns')

        if not np.all(np.abs(mc_sums - 1) <= np.finfo(float).eps):
            raise ValueError('Mixing coefficients must sum to one')

        if not self.dataset_priors.sum() == 1:
            raise ValueError('Dataset priors must sum to one')

        if np.any(self.emission_probabilities < 0) or \
                np.any(self.emission_probabilities > 1):
            raise ValueError('Emission probabilities have to be between 0 and 1')

        if not self._dataset_priors.index.equals(self._mixing_coefficients.index):
            raise ValueError('Dataset priors index does not match mixing coefficients index')

        if not self._mixing_coefficients.columns.equals(self._emission_probabilities.index):
            raise ValueError('Mixing coefficient columns do not match emission probabilities index')

    def __init__(self, dataset_priors, mixing_coefficients, emission_probabilities):

        dataset_priors = pd.Series(dataset_priors)
        self._dataset_priors = dataset_priors

        if isinstance(mixing_coefficients, pd.Series):
            mixing_coefficients = pd.DataFrame(mixing_coefficients).T
        elif isinstance(mixing_coefficients, pd.DataFrame):
            pass
        else:
            mixing_coefficients = np.atleast_2d(mixing_coefficients)
            mixing_coefficients = pd.DataFrame(mixing_coefficients)

        self._mixing_coefficients = mixing_coefficients
        self._emission_probabilities = pd.DataFrame(emission_probabilities)

        self._validate_init()

    def _validate_data(self, data):
        columns = data.columns

        if WEIGHT_COLUMN not in columns:
            raise ValueError('Weight collumn {!r} not found in data columns'.format(WEIGHT_COLUMN))
        elif DATASET_ID_COLUMN not in columns:
            raise ValueError('Dataset id column {!r} not found in data columns'.format(DATASET_ID_COLUMN))

        data_columns = self.data_index
        data_columns_isin = data_columns.isin(data)

        if not data_columns_isin.all():
            not_found = data_columns[~data_columns_isin]
            raise ValueError('Some expected data columns {!r} not in data'.format(not_found))

        dataset_index_unique = data[DATASET_ID_COLUMN].unique()
        dataset_index = self.datasets_index

        if set(dataset_index_unique) != set(dataset_index):
            raise ValueError('Dataset id column does not match the dataset index for mixing coefficients')

        weights = data[WEIGHT_COLUMN]

        if not np.all(weights > 0):
            raise ValueError('Provided weights have to be >0')

    def _to_bool(self, data):

        data = data[self.data_index]
        not_null_mask = ~data.isnull()
        data_as_bool = data.astype(bool)

        return data_as_bool, not_null_mask

    def _support(self, dataset_ids, data_as_bool, not_null_mask):

        pis = self.mixing_coefficients.loc[dataset_ids]
        pis.index = dataset_ids.index

        p = self.emission_probabilities

        support = np.empty((len(data_as_bool), len(pis.columns)),
                           dtype=float,
                           order='F')

        for k_i, k in enumerate(pis.columns):
            pi_k = pis[k]

            ps = partial_support(data_as_bool.values, not_null_mask.values, p.loc[k].values)

            support_k = ps * pi_k
            support[:, k_i] = support_k

        support = pd.DataFrame(support, index=data_as_bool.index, columns=pis.columns)
        return support

    def _individual_log_likelihoods_from_support_log_mus_and_weight(self, support, log_mus, weights):
        support_sum = support.sum(axis=1).apply(np.log)
        return (support_sum + log_mus) * weights

    def _individual_log_likelihoods(self, data):

        dataset_ids = data[DATASET_ID_COLUMN]
        support = self._support(dataset_ids, *self._to_bool(data))
        log_mus = self._log_mus(data)

        return self._individual_log_likelihoods_from_support_log_mus_and_weight(support, log_mus,
                                                                                data[WEIGHT_COLUMN])

    def _log_likelihood_from_support_log_mus_and_weight(self, support, log_mus, weights):
        return self._individual_log_likelihoods_from_support_log_mus_and_weight(support, log_mus, weights).sum()

    def _log_mus(self, data):
        log_mus = self.dataset_priors.loc[data[DATASET_ID_COLUMN]].apply(np.log)
        log_mus.index = data.index

        return log_mus

    def log_likelihood(self, data):
        self._validate_data(data)
        dataset_ids = data[DATASET_ID_COLUMN]
        support = self._support(dataset_ids, *self._to_bool(data))
        log_mus = self._log_mus(data)

        return self._log_likelihood_from_support_log_mus_and_weight(support, log_mus, data[WEIGHT_COLUMN])

    def _mu_update_from_data(self, data):

        counts = data[[DATASET_ID_COLUMN, WEIGHT_COLUMN]].groupby(DATASET_ID_COLUMN).sum()
        counts = counts[WEIGHT_COLUMN]
        total_weight = counts.sum()

        counts /= total_weight

        counts = counts.reindex(self.datasets_index)
        counts.name = self.dataset_priors.name

        return counts

    def _pi_update_from_data(self, data, zstar):

        pi = np.empty(self._mixing_coefficients.shape)

        weights = data[WEIGHT_COLUMN]
        for i, dataset in enumerate(self.datasets_index):

            mask = data[DATASET_ID_COLUMN] == dataset

            sub_weights = weights[mask]
            sub_zstar = zstar[mask]

            ans = sub_zstar.multiply(sub_weights, axis=0).sum(axis=0) / sub_weights.sum()

            pi[i] = ans

        pi = pd.DataFrame(pi, index=self.mixing_coefficients.index, columns=self.mixing_coefficients.columns)
        return pi

    def _p_update_from_data(self, weights, data_as_bool, not_null_mask, zstar):
        old_p = self.emission_probabilities

        zstar_times_weight = zstar.multiply(weights, axis=0)
        zstar_times_weight_sum = zstar_times_weight.sum()

        new_p = p_update_unnormalised(data_as_bool.values, not_null_mask.values,
                                      zstar_times_weight.values, old_p.values)
        new_p = pd.DataFrame(new_p, index=old_p.index, columns=old_p.columns)
        new_p = new_p.divide(zstar_times_weight_sum, axis=0)

        return new_p

    @classmethod
    def collapse_dataset(cls, dataset):

        def _isnan(x):
            try:
                return np.isnan(x)
            except TypeError:
                return x is None

        assert DATASET_ID_COLUMN in dataset.columns
        assert WEIGHT_COLUMN in dataset.columns

        counter = Counter()
        cols = [c for c in dataset.columns if c != WEIGHT_COLUMN]

        for ix, row in dataset.iterrows():
            tuple_row = tuple([x if not _isnan(x) else None for x in row[cols]])
            counter[tuple_row] += row[WEIGHT_COLUMN]

        new_df = []
        for row, weight in counter.items():
            row = row + (weight,)
            new_df.append(row)

        new_df = pd.DataFrame(new_df, columns=cols + [WEIGHT_COLUMN])
        return new_df

    def fit(self, data, n_iter=100, eps=_EPSILON, verbose=True):
        self._validate_data(data)

        dataset_ids = data[DATASET_ID_COLUMN]
        data_as_bool, not_null_mask = self._to_bool(data)

        weights = data[WEIGHT_COLUMN]
        previous_support = self._support(dataset_ids, data_as_bool, not_null_mask)
        log_mus = self._log_mus(data)

        current_log_likelihood = self._log_likelihood_from_support_log_mus_and_weight(previous_support,
                                                                                      log_mus,
                                                                                      weights)

        previous_log_likelihood = current_log_likelihood

        if verbose:
            print('Starting log likelihood: {}'.format(previous_log_likelihood))

        iteration = 0
        converged = False

        while True:
            iteration += 1
            if n_iter is not None and iteration > n_iter:
                break

            if verbose:
                print('Iteration #{}'.format(iteration))

            z_star = previous_support.divide(previous_support.sum(axis=1), axis=0)

            new_pi = self._pi_update_from_data(data, z_star)
            new_p = self._p_update_from_data(weights, data_as_bool, not_null_mask, z_star)

            if iteration == 0:
                new_mu = self._mu_update_from_data(data)
                self._dataset_priors = new_mu
                log_mus = self._log_mus(data)

            self._mixing_coefficients = new_pi
            self._emission_probabilities = new_p

            support = self._support(dataset_ids, data_as_bool, not_null_mask)

            current_log_likelihood = self._log_likelihood_from_support_log_mus_and_weight(support,
                                                                                          log_mus,
                                                                                          weights)

            diff = current_log_likelihood - previous_log_likelihood
            if verbose:
                print('Likelihood {}: (diff: {})'.format(current_log_likelihood, diff))

            assert diff >= -np.finfo(float).eps, \
                'Log likelihood decreased in iteration {}'.format(n_iter)

            if diff <= eps:
                if verbose:
                    print('Converged')
                converged = True
                break

            previous_log_likelihood = current_log_likelihood
            previous_support = support

        return converged, iteration, current_log_likelihood

    @property
    def dataset_priors(self):
        return self._dataset_priors

    @property
    def n_components(self):
        return self._mixing_coefficients.shape[1]

    @property
    def n_datasets(self):
        return len(self.datasets_index)

    @property
    def n_free_parameters(self):

        mu_free_parameter_count = self.n_datasets - 1
        pi_free_parameter_count = self.n_datasets * (self.n_components - 1)

        p_free_parameter_count = self.n_components * self.n_dimensions

        return mu_free_parameter_count + pi_free_parameter_count + p_free_parameter_count

    def BIC(self, log_likelihood, sum_of_weights):
        return -2 * log_likelihood + self.n_free_parameters * np.log(sum_of_weights)

    @property
    def datasets_index(self):
        return self._dataset_priors.index

    @property
    def n_dimensions(self):
        return len(self.data_index)

    @property
    def data_index(self):
        return self.emission_probabilities.columns

    @property
    def mixing_coefficients(self):
        return self._mixing_coefficients

    @property
    def emission_probabilities(self):
        return self._emission_probabilities

