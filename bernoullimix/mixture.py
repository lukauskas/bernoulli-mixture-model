from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Counter

import numpy as np
import pandas as pd

from bernoullimix._mixture import log_support_c, p_update, unnormalised_pi_weights, \
    unnormalised_p_weights, pi_update_c
from cached_property import cached_property
import time

_EPSILON = np.finfo(np.float).eps

DATASET_ID_COLUMN = 'dataset_id'
WEIGHT_COLUMN = 'weight'


def _responsibilities_from_log_support(log_support):
    log_normalisation = np.logaddexp.reduce(log_support, axis=1)
    log_responsibilities = log_support.subtract(log_normalisation, axis=0)
    return np.exp(log_responsibilities)


def _individual_log_likelihoods_from_support_and_weight(log_support,
                                                        weights):
    support_sum = np.logaddexp.reduce(log_support, axis=1)
    # support_sum = log_support.sum(axis=1).apply(np.log)
    return support_sum * weights


def _log_likelihood_from_support_and_weight(log_support, weights):
    return _individual_log_likelihoods_from_support_and_weight(log_support,
                                                               weights).sum()


class MultiDatasetMixtureModel(object):
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

        if not np.allclose(mc_sums, 1.0):
            logger.error('Mixing coefficients do not sum to one. Difference: \n{!r}'.format(
                np.abs(mc_sums - 1)))
            raise ValueError('Mixing coefficients must sum to one')

        if np.any(self.emission_probabilities < 0) or \
                np.any(self.emission_probabilities > 1):
            raise ValueError('Emission probabilities have to be between 0 and 1')

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

    def __init__(self,
                 mixing_coefficients,
                 emission_probabilities,
                 prior_mixing_coefficients=None,
                 prior_emission_probabilities=None):

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
            prior_mixing_coefficients = prior_mixing_coefficients.loc[mixing_coefficients.columns]
            pass
        else:
            prior_mixing_coefficients = pd.Series(prior_mixing_coefficients,
                                                  index=mixing_coefficients.columns)
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
            # Make sure it's two-d
            prior_emission_probabilities = prior_emission_probabilities[['alpha', 'beta']]
        else:
            prior_emission_probabilities = pd.DataFrame(prior_emission_probabilities,
                                                        index=self._emission_probabilities.columns,
                                                        columns=['alpha', 'beta'])

        self._prior_emission_probabilities = prior_emission_probabilities.astype(np.float64)

        self._validate_init()

    def sort_states_by_probabilities(self):
        p = self.emission_probabilities
        pi = self.mixing_coefficients
        p_prior = self.prior_emission_probabilities
        pi_prior = self.prior_mixing_coefficients

        sorted_states = pi.mean(axis=0).order(ascending=False).index

        new_p = p.loc[sorted_states].copy()
        new_p.index = ['S{}'.format(i) for i in range(1, len(p.index) + 1)]

        new_pi = pi[sorted_states].copy()
        new_pi.columns = new_p.index

        new_pi_prior = pi_prior.loc[sorted_states].copy()
        new_pi_prior.index = new_p.index

        return MultiDatasetMixtureModel(new_pi, new_p, new_pi_prior, p_prior)

    def components_to_true_states(self, data, true_states):
        """
        Computes the probability of each of the true state being generate from the component.
        """

        assert data.index.equals(true_states.index)

        responsibilities = self.responsibilities(data)
        weights = data[WEIGHT_COLUMN]

        r_times_weight = responsibilities.multiply(weights, axis=0)
        true_state_priors = true_states.value_counts() / len(true_states)

        component_given_state = {}
        for ts in true_state_priors.index:
            unnormalised = r_times_weight[true_states == ts].sum()
            component_given_state[ts] = unnormalised / unnormalised.sum()

        component_given_state = pd.DataFrame(component_given_state).T
        state_given_component = component_given_state.multiply(true_state_priors, axis=0)
        state_given_component /= state_given_component.sum()
        state_given_component = state_given_component.T

        return component_given_state, state_given_component

    def _validate_data(self, data, allow_not_exact_match_for_dataset=False):
        columns = data.columns

        if WEIGHT_COLUMN not in columns:
            raise ValueError('Weight collumn {!r} not found in data columns'.format(WEIGHT_COLUMN))
        elif DATASET_ID_COLUMN not in columns:
            raise ValueError(
                'Dataset id column {!r} not found in data columns'.format(DATASET_ID_COLUMN))

        dataset_index = set(data[DATASET_ID_COLUMN].unique())

        if not allow_not_exact_match_for_dataset:
            if not frozenset(dataset_index) == frozenset(self.datasets_index):
                raise ValueError('Dataset ids in the data {} do not match dataset ids in model spcification {}'.format(dataset_index,
                                                                                                                        self.datasets_index))
        else:
            if len(frozenset(dataset_index) - frozenset(self.datasets_index)):
                raise ValueError('Dataset ids {} not defined by model'.format(frozenset(dataset_index) - frozenset(self.datasets_index)))

        data_columns = self.data_index
        data_columns_isin = data_columns.isin(data)

        if not data_columns_isin.all():
            not_found = data_columns[~data_columns_isin]
            raise ValueError('Some expected data columns {!r} not in data'.format(not_found))

        weights = data[WEIGHT_COLUMN]

        if not np.all(weights > 0):
            raise ValueError('Provided weights have to be >0')

    def _to_bool(self, data):

        data = data[self.data_index]
        not_null_mask = ~data.isnull()
        # Fill nans as 2
        data_as_bool = data.fillna(2).astype(np.uint8)

        not_null_mask = not_null_mask.astype(np.uint8)
        return data_as_bool, not_null_mask

    def _log_support(self, dataset_ids_as_ilocs, data_as_bool, not_null_mask):
        support = log_support_c(data_as_bool.values, not_null_mask.values,
                                dataset_ids_as_ilocs.values,
                                self.mixing_coefficients.values,
                                self.emission_probabilities.values)

        support = pd.DataFrame(support, index=data_as_bool.index,
                               columns=self.mixing_coefficients.columns)
        return support

    def responsibilities(self, dataset):
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(dataset)
        data_as_bool, not_null_mask = self._to_bool(dataset)
        log_support = self._log_support(dataset_ids_as_ilocs, data_as_bool, not_null_mask)
        return _responsibilities_from_log_support(log_support)

    def _dataset_ids_as_pis_ilocs(self, data):
        dataset_ids = data[DATASET_ID_COLUMN]
        coefs_index = self.mixing_coefficients.index
        return dataset_ids.apply(coefs_index.get_loc)

    def _individual_log_likelihoods(self, data):
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(data)

        log_support = self._log_support(dataset_ids_as_ilocs, *self._to_bool(data))

        return _individual_log_likelihoods_from_support_and_weight(log_support,
                                                                   data[WEIGHT_COLUMN])

    def log_likelihood(self, data):
        self._validate_data(data)
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(data)
        log_support = self._log_support(dataset_ids_as_ilocs, *self._to_bool(data))

        return _log_likelihood_from_support_and_weight(log_support, data[WEIGHT_COLUMN])

    @cached_property
    def _prior_mixing_coefficient_denominator_adjustment(self):
        return self.prior_mixing_coefficients.sum() - self.n_components

    def _dataset_masks(self, data):
        masks = []
        for dataset in self.datasets_index:
            masks.append(data[DATASET_ID_COLUMN] == dataset)
        masks = pd.DataFrame(masks, index=self.datasets_index)
        return masks

    def _encoded_memberships(self, data):
        # Let's hope we will never have >256 datasets?
        return data[DATASET_ID_COLUMN].apply(self.datasets_index.get_loc).astype(np.uint8)

    def _dataset_weight_sums(self, data):
        ans = data.groupby(DATASET_ID_COLUMN)[WEIGHT_COLUMN].sum()
        ans = ans.loc[self.datasets_index].astype(np.float64)

        return ans

    def _pi_update(self, zstar_times_weight, memberships, weight_sums):

        pi = pi_update_c(zstar_times_weight,
                         memberships,
                         weight_sums,
                         self.prior_mixing_coefficients.values,
                         self.n_datasets)

        pi = pd.DataFrame(pi,
                          index=self.mixing_coefficients.index,
                          columns=self.mixing_coefficients.columns)
        return pi

    def _pi_update_from_data(self, data, zstar):

        memberships = self._encoded_memberships(data).values
        weight_sums = self._dataset_weight_sums(data).values

        weights = data[WEIGHT_COLUMN]
        zstar_times_weight = zstar.multiply(weights, axis=0)

        return self._pi_update(zstar_times_weight.values, memberships, weight_sums)

    def _p_update_from_data(self, zstar_times_weight, data_as_bool, not_null_mask):
        old_p = self.emission_probabilities

        # zstar_times_weight_sum = zstar_times_weight.sum()
        p_priors = self.prior_emission_probabilities.values
        new_p = p_update(data_as_bool.values, not_null_mask.values,
                         zstar_times_weight,
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
        pi = self.mixing_coefficients
        pi_weights = unnormalised_pi_weights(pi_prior.values, pi.values)

        p_prior = self.prior_emission_probabilities

        p = self.emission_probabilities

        p_weights = unnormalised_p_weights(p_prior.values, p.values)


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

    def fit(self, data, n_iter=None, eps=_EPSILON, logger=None):

        start_time = time.process_time()

        self._validate_data(data)

        if logger is None:
            logger = self.cls_logger()

        dataset_ids = data[DATASET_ID_COLUMN]
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(data)
        data_as_bool, not_null_mask = self._to_bool(data)

        weights = data[WEIGHT_COLUMN].astype(np.float)

        previous_log_support = self._log_support(dataset_ids_as_ilocs, data_as_bool, not_null_mask)

        current_log_likelihood = _log_likelihood_from_support_and_weight(previous_log_support,
                                                                         weights)

        previous_log_likelihood = current_log_likelihood
        current_posterior = previous_posterior = self._unnormalised_posterior(
            previous_log_likelihood,
            compute_gammas=False)

        _extras = dict(status='started',
                       log_likelihood=current_log_likelihood,
                       posterior=current_posterior)

        logger.debug('BMM: {status}. Log Likelihood: {log_likelihood}. '
                     'Posterior: {posterior}'.format(**_extras),
                     extra=_extras)

        iteration = 0
        converged = False

        DEBUG_EVERY_X_ITERATIONS = 100

        memberships = self._encoded_memberships(data).values
        dataset_weight_sums = self._dataset_weight_sums(data).values

        diff = np.nan

        previous_pi = self.mixing_coefficients
        previous_p = self.emission_probabilities

        time_since_previous_iteration_block = time.time()
        while True:
            if n_iter is not None and iteration >= n_iter:
                break
            iteration += 1

            zstar = _responsibilities_from_log_support(previous_log_support)
            zstar_times_weight = zstar.multiply(weights.values, axis=0).values

            new_pi = self._pi_update(zstar_times_weight, memberships, dataset_weight_sums)
            new_p = self._p_update_from_data(zstar_times_weight, data_as_bool, not_null_mask)

            self._mixing_coefficients = new_pi
            self._emission_probabilities = new_p

            log_support = self._log_support(dataset_ids_as_ilocs, data_as_bool, not_null_mask)

            current_log_likelihood = _log_likelihood_from_support_and_weight(log_support,
                                                                             weights)

            current_posterior = self._unnormalised_posterior(current_log_likelihood,
                                                             compute_gammas=False)

            diff = current_posterior - previous_posterior

            try:
                assert diff >= -np.finfo(float).eps, \
                    'Unnormalised posterior decreased in iteration {}. Change {} -> {}'.format(
                        iteration, previous_posterior, current_posterior)
            except AssertionError:
                logger.debug(
                    'Parameters before:\nPI:\n{!r}\nP:\n{!r}'.format(previous_pi, previous_p))

                logger.debug(
                    'Parameters after:\n\nPI:\n{!r}\nP:\n{!r}'.format(self.mixing_coefficients,
                                                                      self.emission_probabilities))

                logger.debug(
                    'LL change: {} -> {}'.format(previous_log_likelihood, current_log_likelihood,
                                                 current_log_likelihood - previous_log_likelihood))
                logger.debug(
                    'Posterior change: {} -> {} ({})'.format(previous_posterior, current_posterior,
                                                             diff))
                raise

            if diff <= eps:
                converged = True
                break

            if iteration % DEBUG_EVERY_X_ITERATIONS == 0:
                current_time = time.process_time()
                duration_iteration = current_time - time_since_previous_iteration_block
                _extras = dict(status='running',
                               log_likelihood=current_log_likelihood,
                               posterior=current_posterior,
                               iteration=iteration,
                               duration=duration_iteration,
                               diff=diff)
                logger.debug('BMM: {status}. Iteration: {iteration}. Took: {duration:.2f} seconds '
                             'Log Likelihood: {log_likelihood}. '
                             'Posterior: {posterior}. Last Diff: {diff}'.format(**_extras),
                             extra=_extras)
                time_since_previous_iteration_block = time.process_time()

            previous_log_likelihood = current_log_likelihood
            previous_posterior = current_posterior
            previous_log_support = log_support
            previous_pi = self.mixing_coefficients
            previous_p = self.emission_probabilities

        end_time = time.process_time()
        duration_seconds = (end_time - start_time)

        _extras = dict(status='finished',
                       converged='yes' if converged else 'no',
                       log_likelihood=current_log_likelihood,
                       posterior=current_posterior,
                       iteration=iteration,
                       diff=diff,
                       duration_seconds=duration_seconds)

        logger.debug('BMM: {status}. Iteration: {iteration}. '
                     'Converged: {converged}. Took: {duration_seconds}s. '
                     'Log Likelihood: {log_likelihood}. '
                     'Posterior: {posterior}. '
                     'Last Diff: {diff} '.format(**_extras),
                     extra=_extras)

        return converged, iteration, current_log_likelihood


    @property
    def n_components(self):
        return self._mixing_coefficients.shape[1]

    @property
    def n_datasets(self):
        return len(self.datasets_index)

    @classmethod
    def _mle_states(cls, log_support):
        """
        Computes the most likely states to have generated each of the data points
        from the provided log support

        :param log_support: N x K matrix, suggesting the support from state k for each of the N datapoints
        :return:
        """
        return log_support.idxmax(axis=1)

    def mle_states(self, dataset):
        """
        Returns most likely states to have generated each of the dataset points from the model

        :param dataset: dataset
        :return:
        """
        self._validate_data(dataset, allow_not_exact_match_for_dataset=True)
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(dataset)
        log_support = self._log_support(dataset_ids_as_ilocs, *self._to_bool(dataset))
        return self._mle_states(log_support)

    def complete_mle_log_likelihood(self, dataset):
        self._validate_data(dataset, allow_not_exact_match_for_dataset=True)
        dataset_ids_as_ilocs = self._dataset_ids_as_pis_ilocs(dataset)

        bool_data, not_null_mask = self._to_bool(dataset)
        log_support = self._log_support(dataset_ids_as_ilocs, bool_data, not_null_mask)

        # First part of the data (observed given MLE state)
        ans = log_support.max(axis=1)

        # Second part of the data (contribution from hidden datapoints)
        states = self._mle_states(log_support)

        hidden_p_additions = self.emission_probabilities

        # mle estimate for x_i,d given state is 1 if p_k,d >= 0.5 else 0
        # since we will multiply that by log probability almost immediately, we do it here:
        hidden_p_additions = hidden_p_additions.applymap(lambda p: np.log(p) if p >= 0.5 else np.log(1-p))
        # then the hidden components only contribute the above where the data is null
        hidden_p_additions = hidden_p_additions.loc[states]
        # Replace index
        hidden_p_additions.index = not_null_mask.index
        ans += hidden_p_additions[~(not_null_mask.astype(bool))].sum(axis=1)

        # Remember the weights!
        ans *= dataset[WEIGHT_COLUMN]
        return ans.sum()



    @property
    def n_free_parameters(self):

        pi_free_parameter_count = self.n_datasets * (self.n_components - 1)

        p_free_parameter_count = self.n_components * self.n_dimensions

        return pi_free_parameter_count + p_free_parameter_count

    def BIC(self, log_likelihood, sum_of_weights):
        return -2 * log_likelihood + self.n_free_parameters * np.log(sum_of_weights)

    def ICL(self, dataset):
        ll = self.complete_mle_log_likelihood(dataset)
        return -2 * ll + self.n_free_parameters * np.log(dataset[WEIGHT_COLUMN].sum())

    @property
    def datasets_index(self):
        return self._mixing_coefficients.index

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
        return self.mixing_coefficients.equals(other.mixing_coefficients) \
               and self.emission_probabilities.equals(other.emission_probabilities)
