from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Counter

import numpy as np
import pandas as pd

from bernoullimix._bernoulli import probability_z_o_given_theta_c, \
    _log_likelihood_from_z_o_joint, _posterior_probability_of_class_given_support, _m_step, _em, \
    impute_missing_data_c

_EPSILON = np.finfo(np.float).eps


class MixtureModel(object):
    _mixing_coefficients = None
    _emission_probabilities = None

    def _validate(self):
        if not self.mixing_coefficients.columns.equals(self.emission_probabilities.index):
            raise ValueError('The mixing coefficients index does not match emission probabilities '
                             'index {!r} != {!r}'.format(self.mixing_coefficients.columns,
                                                         self.emission_probabilities.index))

        mc_sums = self.mixing_coefficients.sum(axis='columns')

        if not np.all(mc_sums == 1):
            raise ValueError('Mixing coefficients must sum to one')

        if np.any(self.emission_probabilities < 0) or \
                np.any(self.emission_probabilities > 1):
            raise ValueError('Emission probabilities have to be between 0 and 1')

    def __init__(self, mixing_coefficients, emission_probabilities):
        if isinstance(mixing_coefficients, pd.Series):
            mixing_coefficients = pd.DataFrame(mixing_coefficients).T
        elif isinstance(mixing_coefficients, pd.DataFrame):
            pass
        else:
            mixing_coefficients = np.atleast_2d(mixing_coefficients)
            mixing_coefficients = pd.DataFrame(mixing_coefficients)

        self._mixing_coefficients = mixing_coefficients
        self._emission_probabilities = pd.DataFrame(emission_probabilities)

        self._validate()

    def log_likelihood(self, datasets, weights=None):

        if not isinstance(datasets, list):
            datasets = [datasets]

        datasets = [pd.DataFrame(d) for d in datasets]

        if len(datasets) != self.n_datasets:
            raise ValueError('{} datasets provided, but expected {}'.format(len(datasets),
                                                                            self.n_datasets))

        for i, dataset in enumerate(datasets):
            if not dataset.columns.equals(self.emission_probabilities.columns):
                raise ValueError('Dataset #{} columns do not '
                                 'match the emission probability columns:\n'
                                 '{!r} != {!r}'.format(i, dataset.columns,
                                                       self.emission_probabilities.columns))

        if weights is None:
            weights = []

            for d in datasets:
                w = pd.Series(np.ones(len(d)), index=d.index)
                weights.append(w)

        if len(weights) != self.n_datasets:
            raise ValueError('{} weights provided, but expected {}'.format(len(weights),
                                                                           self.n_datasets))
        for i, (w, d) in enumerate(zip(weights, datasets)):
            if not d.index.equals(w.index):
                raise ValueError('Dataset #{} index does not match the corresponding weights index')

    @property
    def n_components(self):
        return self._mixing_coefficients.shape[1]

    @property
    def n_datasets(self):
        return self._mixing_coefficients.shape[0]

    @property
    def n_dimensions(self):
        return self._emission_probabilities.shape[1]

    @property
    def mixing_coefficients(self):
        return self._mixing_coefficients

    @property
    def emission_probabilities(self):
        return self._emission_probabilities



class ConvergenceStatus(object):

    converged = None
    number_of_iterations = None
    likelihood_trace = None

    def __init__(self, converged, number_of_iterations, likelihood_trace=None):
        self.converged = converged
        self.number_of_iterations = number_of_iterations
        self.likelihood_trace = likelihood_trace

    @property
    def trace_available(self):
        return self.likelihood_trace is not None

    def __repr__(self):
        converged_text = 'converged' if self.converged else 'did not converge'
        trace_text = ' (trace available)' if self.trace_available else ''
        return '<{} in {:,} iterations{}>'.format(converged_text, self.number_of_iterations,
                                                  trace_text)


class BernoulliMixture(object):
    _number_of_components = None
    _number_of_dimensions = None

    _mixing_coefficients = None
    _emission_probabilities = None

    def __init__(self, mixing_coefficients, emission_probabilities):
        """
        Initialises a Bernoulli Mixture Model

        :param mixing_coefficients: K-dimensional array of the mixture components for the data
        :param emission_probabilities: (K, D)-dimensional matrix of the probabilities of emitting
                                       `True` in each bernoulli, given the k.
        """
        self._mixing_coefficients = np.asarray(mixing_coefficients, dtype=float)
        self._emission_probabilities = np.asarray(emission_probabilities, dtype=float)

        self._number_of_components = self._mixing_coefficients.shape[0]
        self._number_of_dimensions = self._emission_probabilities.shape[1]

        self._validate()

    def _bounded_between_zero_and_one(self, array_):
        return np.all((array_ >= 0) & (array_ <= 1))

    def _validate(self):
        K = self.number_of_components
        D = self.number_of_dimensions

        if not self.mixing_coefficients.shape == (K, ):
            raise ValueError('Wrong shape of mixing coefficients provided. '
                             'Expected {}, got {}'.format((K,), self.mixing_coefficients))

        if not np.isclose(np.sum(self.mixing_coefficients), 1.0):
            raise ValueError('Mixing coefficient probabilities do not sum to one. Got: {}'.format(
                np.sum(self.mixing_coefficients)))

        if not self._bounded_between_zero_and_one(self.mixing_coefficients):
            raise ValueError('Mixing coefficients not bounded between 0 and 1')

        if not self.emission_probabilities.shape == (K, D):
            raise ValueError('Wrong shape of emission probabilities matrix. '
                             'Expected {}, got {}'.format((K, D), self.emission_probabilities.shape))

        if not self._bounded_between_zero_and_one(self.emission_probabilities):
            raise ValueError('Emission probabilities not bounded between 0 and 1')

    @property
    def number_of_components(self):
        return self._number_of_components

    @property
    def number_of_dimensions(self):
        return self._number_of_dimensions

    @property
    def emission_probabilities(self):
        return self._emission_probabilities

    @property
    def mixing_coefficients(self):
        return self._mixing_coefficients

    @property
    def number_of_free_parameters(self):
        """
        Returns number of free parameters for module
        :return:
        """
        # K - 1 params for mixture components
        # (K * D) parameters for emission probabilities

        return (self.number_of_components - 1) + \
               (self.number_of_components * self.number_of_dimensions)

    @classmethod
    def aggregate_dataset(cls, dataset):
        """
        Take the dataset and return only its unique rows, along with their counts

        :param dataset: dataset to process
        :return: tuple. First element is the unique rows in data (pd.DataFrame),
                        second element is the number of times they occur
        """
        # This is required to work with pandas DataFrames sometimes
        dataset = pd.DataFrame(dataset)

        def _hash(row):
            return tuple([int(x) if x is not None and not np.isnan(x) else None for x in row])

        counts = Counter(dataset.apply(_hash, axis=1))

        unique = dataset.drop_duplicates()
        counts = unique.apply(lambda row: counts.get(_hash(row)), axis=1)

        return unique, counts

    def _penalised_likelihood(self, log_likelihood, psi):
        """
        Returns penalised likelihood computed as:

        $$
            -2L + \psi \eta
        $$
        Where $L$ is the log likelihood (provided),
        $\eta$ is the number of free parameters in the model,
        and $\psi$ is the provided penalty term.
        For instance set psi=2 to get AIC, or psi=log N to get BIC.

        :param log_likelihood: log likelihood
        :param psi: penalty term
        :return: penalised likelihood
        """
        return -2.0 * log_likelihood + psi * self.number_of_free_parameters

    def BIC_dataset(self, dataset):
        """
        Computes Bayesian Information Criterion for Dataset
        :param dataset: dataset to compute BIC for
        :return: BIC
        """
        log_likelihood = self.log_likelihood(dataset)
        return self.BIC(log_likelihood, len(dataset))

    def BIC(self, log_likelihood, number_of_observations):
        """
        Computes BIC given log likelihood and number of observations
        :param log_likelihood:
        :param number_of_observations:
        :return:
        """
        psi = np.log(number_of_observations)
        return self._penalised_likelihood(log_likelihood, psi=psi)

    def AIC_dataset(self, dataset):
        """
        Computes Akaike Information Criterion for dataset
        :param dataset: dataset to compute AIC for
        :return: AIC
        """
        log_likelihood = self.log_likelihood(dataset)
        return self.AIC(log_likelihood)

    def AIC(self, log_likelihood):
        """
        Computes Akaike Information Criterion for log likelihood
        :param log_likelihood:
        :return:
        """
        psi = 2
        return self._penalised_likelihood(log_likelihood, psi=psi)

    def sample(self, size, random_state=None):
        """
        Sample a `size` amount of observations from mixture model.

        :param size: the number of observations to sample
        :param random_state: (optional) random state to use.
        :return: (observations, true_components) -- two arrays. The generated observations and their
                true components.
        """
        random = np.random.RandomState(random_state)

        true_components = np.argmax(random.multinomial(1, self.mixing_coefficients, size=size),
                                    axis=1)
        observations = np.empty((size, self.number_of_dimensions))

        for component in range(self.number_of_components):
            mask = true_components == component
            n_samples_for_component = np.sum(true_components == component)

            for dimension in range(self.number_of_dimensions):
                prob = self.emission_probabilities[component, dimension]

                samples_for_component = random.binomial(1,
                                                        prob,
                                                        size=n_samples_for_component)

                observations[mask, dimension] = samples_for_component

        return observations, true_components

    def log_likelihood(self, dataset):

        if dataset.shape[1] != self.number_of_dimensions:
            raise ValueError('The dataset shape does not match number of dimensions.'
                             'Got {}, expected {}'.format(dataset.shape[1],
                                                          self.number_of_dimensions))

        unique_dataset, weights = self.aggregate_dataset(dataset)
        support = self._prob_z_o_given_theta(unique_dataset)
        return self._log_likelihood_from_support(support, weights)

    def _log_likelihood_from_support(self, support, weights):
        """
        Computes log likelihood from the support

        :param support: support (computed by `BernoulliMixture._observation_emission_support`)
        :param weights: weights for each support row (i.e. how many rows does it represent)
        :return:
        """
        support = np.asarray(support)
        weights = np.asarray(weights)

        return _log_likelihood_from_z_o_joint(support, weights)

    def _prob_z_o_given_theta(self, observations):
        """
        Returns point emission probabilities for a set of observations provided as array.
        Usually the code would just compute it for unique observations, and then
        weigh the support appropriately in calculations.
        :param observations: array of observations
        """

        observations, mask = self._as_decoupled_array(observations)

        return probability_z_o_given_theta_c(observations, mask, self.emission_probabilities,
                                             self.mixing_coefficients)

    @classmethod
    def _posterior_probability_of_class_given_support(cls, support):
        return _posterior_probability_of_class_given_support(support)


    def fit(self, dataset, iteration_limit=1000, convergence_threshold=_EPSILON,
            trace_likelihood=False):
        """
        Fits the mixture model to the dataset using EM algorithm.

        :param dataset: dataset to fit to
        :param iteration_limit: number of iterations to search. If none, will run till convergence
        :param convergence_threshold: threshold (for log likelihood) that marks convergence
        :param trace_likelihood: if set to true, the likelihood trace from optimisation
                                 will be returned
        :return: (float, `ConvergenceStatus`) : log likelihood of the dataset post fitting,
            and the information about convergence of the algorithm
        """

        dataset = pd.DataFrame(dataset)

        if dataset.shape[1] != self.number_of_dimensions:
            raise ValueError('The dataset shape does not match number of dimensions.'
                             'Got {}, expected {}'.format(dataset.shape[1],
                                                          self.number_of_dimensions))

        # Get only unique rows and their counts
        unique_dataset, counts = self.aggregate_dataset(dataset)

        return self.fit_aggregated(unique_dataset, counts, iteration_limit, convergence_threshold,
                                   trace_likelihood)

    def fit_aggregated(self, unique_dataset, counts,
                       iteration_limit=1000,
                       convergence_threshold=_EPSILON,
                       trace_likelihood=False):
        """
        Fits the mixture model to the dataset using EM algorithm.
        Same as `fit()`, but takes aggregated dataset as input.

        :param unique_dataset: dataset to fit to (aggregated)
        :param counts: counts for each of the rows (as returned by the aggregate dataset function)
         :param iteration_limit: number of iterations to search. If none, will run till convergence
        :param convergence_threshold: threshold (for log likelihood) that marks convergence
        :param trace_likelihood: if set to true, the likelihood trace from optimisation
                                 will be returned
        :return: (float, `ConvergenceStatus`) : log likelihood of the dataset post fitting,
            and the information about convergence of the algorithm
        """

        assert unique_dataset.index.equals(counts.index)

        mixing_coefficients, emission_probabilities, \
        converged, current_log_likelihood, iterations_done, likelihood_trace = self._em(
            unique_dataset, counts, iteration_limit, convergence_threshold, trace_likelihood)
        self._mixing_coefficients = mixing_coefficients
        self._emission_probabilities = emission_probabilities

        try:
            self._validate()
        except Exception as e:
            raise Exception('EM algorithm converged to invalid state: {}'.format(e))

        convergence_status = ConvergenceStatus(bool(converged), iterations_done, likelihood_trace)
        return current_log_likelihood, convergence_status

    def _em(self, unique_dataset, counts, iteration_limit, convergence_threshold, trace_likelihood):
        unique_dataset_as_array, mask = self._as_decoupled_array(unique_dataset)
        counts_as_array = np.asarray(counts)

        return _em(unique_dataset_as_array, counts_as_array,
                   mask,
                   self.mixing_coefficients, self.emission_probabilities,
                   -1 if iteration_limit is None else iteration_limit,
                   convergence_threshold, 1 if trace_likelihood else 0)

    def soft_assignment(self, dataset):
        """
        Returns soft assignment of dataset to classes given the model.

        :param dataset: Dataset to assign
        :return: (N, K) matrix of probabilities of the n-th observation comming from k K
        """

        support = self._prob_z_o_given_theta(dataset)
        probs = self._posterior_probability_of_class_given_support(support)

        return probs

    def hard_assignment(self, dataset):
        """
        Returns hard assignment of dataset to classes given the model

        :param dataset: Dataset to assign
        :return: N-vector of the most-likely k to generate that vector
        """

        probs = self.soft_assignment(dataset)
        return np.argmax(probs, axis=1)

    @classmethod
    def _as_decoupled_array(cls, dataset):

        dataset = pd.DataFrame(dataset)

        mask = np.asarray(~dataset.isnull(), dtype=bool)
        dataset = np.asarray(dataset, dtype=bool)

        return dataset, mask

    def impute_missing_values(self, dataset):
        """
        Imputes missing values for the dataset
        :param dataset:
        :return:
        """
        dataset = pd.DataFrame(dataset)

        array, mask = self._as_decoupled_array(dataset)
        imputed = impute_missing_data_c(array, mask, self.emission_probabilities,
                                        self.mixing_coefficients)
        imputed = pd.DataFrame(imputed, index=dataset.index, columns=dataset.columns)

        return imputed

