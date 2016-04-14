from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from bernoullimix._mixture import support_c, p_update
from cached_property import cached_property

_EPSILON = np.finfo(np.float).eps

DATASET_ID_COLUMN = 'dataset_id'
WEIGHT_COLUMN = 'weight'


class MultiDatasetMixtureModel(object):

    _dataset_priors = None
    _mixing_coefficients = None
    _emission_probabilities = None

    _prior_mixing_coefficients = None
    _prior_emission_probabilities = None

    def _validate_init(self):
        logger = self.cls_logger()

        if not self.mixing_coefficients.columns.equals(self.emission_probabilities.index):
            raise ValueError('The mixing coefficients index does not match emission probabilities '
                             'index {!r} != {!r}'.format(self.mixing_coefficients.columns,
                                                         self.emission_probabilities.index))

        mc_sums = self.mixing_coefficients.sum(axis='columns')

        if not np.all(np.abs(mc_sums - 1) <= 10 * np.finfo(float).resolution):
            logger.error('Mixing coefficients do not sum to one. Difference: \n{!r}'.format(
                np.abs(mc_sums - 1)))
            raise ValueError('Mixing coefficients must sum to one')

        if np.abs(self.dataset_priors.sum() - 1) > 10 * np.finfo(float).resolution:
            raise ValueError('Dataset priors must sum to one.\n{!r}'.format(self.dataset_priors))

        if np.any(self.emission_probabilities < 0) or \
                np.any(self.emission_probabilities > 1):
            raise ValueError('Emission probabilities have to be between 0 and 1')

        if not self._dataset_priors.index.equals(self._mixing_coefficients.index):
            raise ValueError('Dataset priors index does not match mixing coefficients index')

        if not self._mixing_coefficients.columns.equals(self._emission_probabilities.index):
            raise ValueError('Mixing coefficient columns do not match emission probabilities index')

        if not self._prior_mixing_coefficients.index.equals(self._mixing_coefficients.columns):
            raise ValueError('Mixing coefficient columns must equal the index of their priors')

        if list(self._prior_emission_probabilities.columns) != ['alpha', 'beta']:
            raise ValueError('Prior emission probabilities should have alpha and beta for columns')

        if not self._prior_emission_probabilities.index.equals(
                self._emission_probabilities.columns):
            raise ValueError(
                'Prior emission probabilities index should be the same as emission probs')

    def __init__(self, dataset_priors, mixing_coefficients, emission_probabilities,
                 prior_mixing_coefficients=None,
                 prior_emission_probabilities=None):

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

        if prior_mixing_coefficients is None:
            # dirichlet prior of one is the same as having no dirichlet prior
            prior_mixing_coefficients = pd.Series(1, index=mixing_coefficients.columns)
        elif isinstance(prior_mixing_coefficients, pd.Series):
            pass
        else:
            prior_mixing_coefficients = pd.Series(prior_mixing_coefficients,
                                                  index=self._mixing_coefficients.columns)
        self._prior_mixing_coefficients = prior_mixing_coefficients.astype(np.float64)

        if prior_emission_probabilities is None:
            prior_emission_probabilities = pd.DataFrame(
                np.ones((self._emission_probabilities.shape[1], 2)),
                index=self._emission_probabilities.columns,
                columns=['alpha', 'beta'])
        elif (isinstance(prior_emission_probabilities, list) or isinstance(
                prior_emission_probabilities, tuple)) and len(prior_emission_probabilities) == 2:
            _alpha, _beta = prior_emission_probabilities

            prior_emission_probabilities = pd.concat([pd.Series(_alpha,
                                                                index=self._emission_probabilities.columns,
                                                                name='alpha'),
                                                      pd.Series(_beta,
                                                                index=self._emission_probabilities.columns,
                                                                name='beta'),
                                                      ], axis=1)
        elif isinstance(prior_emission_probabilities, pd.DataFrame):
            pass
        else:
            prior_emission_probabilities = pd.DataFrame(prior_emission_probabilities,
                                                        index=self._emission_probabilities.columns,
                                                        columns=['alpha', 'beta'])

        self._prior_emission_probabilities = prior_emission_probabilities.astype(np.float64)

        self._validate_init()

    def sort_states_by_probabilities(self):
        p = self.emission_probabilities
        pi = self.mixing_coefficients
        mu = self.dataset_priors
        p_prior = self.prior_emission_probabilities
        pi_prior = self.prior_mixing_coefficients

        sorted_states = p.mean(axis=1).order(ascending=False).index

        new_p = p.loc[sorted_states].copy()
        new_p.index = ['S{}'.format(i) for i in range(1, len(p.index) + 1)]

        new_pi = pi[sorted_states].copy()
        new_pi.columns = new_p.index

        new_pi_prior = pi_prior.loc[sorted_states].copy()
        new_pi_prior.index = new_p.index

        return MultiDatasetMixtureModel(mu, new_pi, new_p, new_pi_prior, p_prior)

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

    def _support(self, dataset_ids_as_ilocs, data_as_bool, not_null_mask):
        support = support_c(data_as_bool.values, not_null_mask.values,
                            dataset_ids_as_ilocs.values,
                            self.mixing_coefficients.values,
                            self.emission_probabilities.values)

        support = pd.DataFrame(support, index=data_as_bool.index, columns=self.mixing_coefficients.columns)
        return support

    def responsibilities(self, dataset):
        # TODO: make sure to figure out what happens when we do not know the dataset_ids

        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(dataset)
        data_as_bool, not_null_mask = self._to_bool(dataset)
        support = self._support(dataset_ids_as_ilocs, data_as_bool, not_null_mask)
        responsibilities = support.divide(support.sum(axis=1), axis=0)

        return responsibilities




    def _individual_log_likelihoods_from_support_log_mus_and_weight(self, support, log_mus, weights):
        support_sum = support.sum(axis=1).apply(np.log)
        return (support_sum + log_mus) * weights

    def _dataset_ids_as_pis_ilocs(self, data):
        dataset_ids = data[DATASET_ID_COLUMN]
        coefs_index = self.mixing_coefficients.index
        return dataset_ids.apply(coefs_index.get_loc)

    def _individual_log_likelihoods(self, data):
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(data)

        support = self._support(dataset_ids_as_ilocs, *self._to_bool(data))
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
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(data)
        support = self._support(dataset_ids_as_ilocs, *self._to_bool(data))
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

    @cached_property
    def _prior_mixing_coefficient_denominator_adjustment(self):
        return self.prior_mixing_coefficients.sum() - self.n_components

    def _dataset_masks(self, data):
        masks = {}
        for dataset in self.datasets_index:
            masks[dataset] = data[DATASET_ID_COLUMN] == dataset
        return masks

    def _dataset_weight_sums(self, data, masks=None):
        if masks is None:
            masks = self._dataset_masks(data)
        return pd.Series([data[masks[dataset]][WEIGHT_COLUMN].sum()
                          for dataset in self.datasets_index],
                         index=self.datasets_index)

    def _pi_update(self, zstar_times_weight, masks, weight_sums):

        pi = np.empty(self._mixing_coefficients.shape)
        pi_priors_minus_one = self.prior_mixing_coefficients - 1
        pi_prior_adjustment = self._prior_mixing_coefficient_denominator_adjustment

        for i, dataset in enumerate(self.datasets_index):
            mask = masks[dataset]

            ans = zstar_times_weight[mask].sum(axis=0)
            ans += pi_priors_minus_one
            ans /= weight_sums.loc[dataset] + pi_prior_adjustment

            pi[i] = ans

        pi = pd.DataFrame(pi, index=self.mixing_coefficients.index,
                          columns=self.mixing_coefficients.columns)
        return pi

    def _pi_update_from_data(self, data, zstar):

        masks = self._dataset_masks(data)
        weight_sums = self._dataset_weight_sums(data, masks)

        weights = data[WEIGHT_COLUMN]
        zstar_times_weight = zstar.multiply(weights, axis=0)

        return self._pi_update(zstar_times_weight, masks, weight_sums)

    def _p_update_from_data(self, zstar_times_weight, data_as_bool, not_null_mask):
        old_p = self.emission_probabilities

        # zstar_times_weight_sum = zstar_times_weight.sum()
        p_priors = self.prior_emission_probabilities.values
        new_p = p_update(data_as_bool.values, not_null_mask.values,
                         zstar_times_weight.values,
                         old_p.values,
                         p_priors)
        new_p = pd.DataFrame(new_p, index=old_p.index, columns=old_p.columns)
        # new_p = new_p.divide(zstar_times_weight_sum, axis=0)

        return new_p

    def _unnormalised_posterior(self, log_likelihood, compute_gammas=False):
        """
        Takes log likelihood and multiplies it by the unnomralised priors.
        Used in EM estimation to track change in probability.
        """

        pi_prior = self.prior_mixing_coefficients
        p_prior = self.prior_emission_probabilities

        pi = self.mixing_coefficients
        p = self.emission_probabilities

        pi_weights = ((pi_prior - 1) * pi.apply(np.log)).sum().sum()

        p_weights = p.apply(np.log) * (p_prior['alpha'] - 1)
        p_weights += (1-p).apply(np.log) * (p_prior['beta'] - 1)
        p_weights = p_weights.sum().sum()

        weighted_log_likelihood = log_likelihood + pi_weights + p_weights

        if compute_gammas:
            from scipy.special import gammaln
            K = self.n_components
            T = self.n_datasets

            pi_gamma = T * (gammaln(pi_prior.sum()) - pi_prior.apply(gammaln).sum())
            p_gamma = K * (gammaln(p_prior.sum(axis=1))
                           - gammaln(p_prior['alpha'])
                           - gammaln(p_prior['beta'])).sum()

            weighted_log_likelihood += pi_gamma + p_gamma

        return weighted_log_likelihood


    @classmethod
    def collapse_dataset(cls, dataset, sort_results=False):

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
        if sort_results:
            new_df.sort(columns=[WEIGHT_COLUMN] + cols,
                        ascending=[False] + [True] * len(cols),
                        inplace=True)

        return new_df

    @classmethod
    def cls_logger(cls):
        import logging
        return logging.getLogger('bernoullimix.mixture.MultiDatasetMixtureModel')

    def fit(self, data, n_iter=100, eps=_EPSILON, logger=None):
        self._validate_data(data)

        if logger is None:
            logger = self.cls_logger()

        dataset_ids = data[DATASET_ID_COLUMN]
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(data)
        data_as_bool, not_null_mask = self._to_bool(data)

        weights = data[WEIGHT_COLUMN].astype(np.float)

        previous_support = self._support(dataset_ids_as_ilocs, data_as_bool, not_null_mask)
        log_mus = self._log_mus(data)

        current_log_likelihood = self._log_likelihood_from_support_log_mus_and_weight(previous_support,
                                                                                      log_mus,
                                                                                      weights)

        previous_log_likelihood = current_log_likelihood
        previous_posterior = self._unnormalised_posterior(previous_log_likelihood,
                                                                       compute_gammas=False)

        logger.debug('Starting mu:\n{!r}'.format(self.dataset_priors))
        logger.debug('Starting pi:\n{!r}'.format(self.mixing_coefficients))
        logger.debug('Starting p:\n{!r}'.format(self.emission_probabilities))

        logger.debug('Starting log likelihood: {}'.format(previous_log_likelihood))
        logger.debug('Starting unnormalised posterior: {}'.format(previous_posterior))

        iteration = 0
        converged = False

        DEBUG_EVERY_X_ITERATIONS = 100

        masks = self._dataset_masks(data)
        dataset_weight_sums = self._dataset_weight_sums(data, masks=masks)

        while True:
            if n_iter is not None and iteration >= n_iter:
                break
            iteration += 1

            zstar = previous_support.divide(previous_support.sum(axis=1), axis=0)
            zstar_times_weight = zstar.multiply(weights, axis=0)

            new_pi = self._pi_update(zstar_times_weight, masks, dataset_weight_sums)
            new_p = self._p_update_from_data(zstar_times_weight, data_as_bool, not_null_mask)

            if iteration == 0:
                new_mu = self._mu_update_from_data(data)
                self._dataset_priors = new_mu
                log_mus = self._log_mus(data)

            self._mixing_coefficients = new_pi
            self._emission_probabilities = new_p

            support = self._support(dataset_ids_as_ilocs, data_as_bool, not_null_mask)

            current_log_likelihood = self._log_likelihood_from_support_log_mus_and_weight(support,
                                                                                          log_mus,
                                                                                          weights)

            current_posterior = self._unnormalised_posterior(current_log_likelihood,
                                                             compute_gammas=False)

            diff = current_posterior - previous_posterior
            if iteration % DEBUG_EVERY_X_ITERATIONS == 0:
                logger.debug('Iteration #{}. Likelihood: {}. Posterior: {}'
                             '(diff: {})'.format(iteration, current_log_likelihood, current_posterior, diff))

            assert diff >= -np.finfo(float).eps, \
                'Unnormalised posterior decreased in iteration {}. Difference: {}'.format(n_iter, diff)

            if diff <= eps:
                logger.debug('Converged at iteration {}'.format(iteration))
                converged = True
                break

            previous_log_likelihood = current_log_likelihood
            previous_posterior = current_posterior

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

    @property
    def prior_mixing_coefficients(self):
        return self._prior_mixing_coefficients

    @property
    def prior_emission_probabilities(self):
        return self._prior_emission_probabilities

    def __eq__(self, other):
        return self.dataset_priors.equals(other.dataset_priors) \
               and self.mixing_coefficients.equals(other.mixing_coefficients) \
               and self.emission_probabilities.equals(other.emission_probabilities)
