from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import itertools
import multiprocessing

from bernoullimix import MultiDatasetMixtureModel
from bernoullimix.mixture import WEIGHT_COLUMN
from bernoullimix.random_initialisation import random_mixture_generator
import pandas as pd

def _initializer(data, random_state, n_mixtures_to_search, fit_kwargs,
                 prior_mixing_coefficients, prior_emission_probabilities):
    global g_data, g_random_state, g_n_mixtures_to_search, g_fit_kwargs, \
        g_prior_mixing_coefficients, g_prior_emission_probabilities

    n_mixtures_to_search = int(n_mixtures_to_search)
    g_n_mixtures_to_search = n_mixtures_to_search

    g_data = data
    g_random_state = random_state
    g_fit_kwargs = fit_kwargs

    g_prior_mixing_coefficients = prior_mixing_coefficients
    g_prior_emission_probabilities = prior_emission_probabilities


def _map_function(k, n_mixtures_to_search, data,
                  fit_kwargs,
                  random_state,
                  prior_mixing_coefficients, prior_emission_probabilities):
    generator = random_mixture_generator(k, data,
                                         random_state=random_state,
                                         prior_emission_probabilities=prior_emission_probabilities,
                                         prior_mixing_coefficients=prior_mixing_coefficients)

    mixtures = list(itertools.islice(generator, n_mixtures_to_search))

    best_result = None
    best_mixture = None

    for mixture in mixtures:
        result = mixture.fit(data, **fit_kwargs)
        result = pd.Series(result, index=['converged', 'n_iterations', 'log_likelihood'])
        if best_result is None or result['log_likelihood'] > best_result['log_likelihood']:
            best_result = result
            best_mixture = mixture

    best_result['BIC'] = best_mixture.BIC(best_result['log_likelihood'],
                                          data[WEIGHT_COLUMN].sum())

    return best_result, best_mixture


def _globalised_map_function(k):
    global g_data, g_random_state, g_n_mixtures_to_search, g_fit_kwargs, \
        g_prior_mixing_coefficients, g_prior_emission_probabilities

    return _map_function(k, g_n_mixtures_to_search, g_data,
                         g_fit_kwargs,
                         g_random_state,
                         g_prior_mixing_coefficients, g_prior_emission_probabilities)


def search_k(k_range_to_search, data, mixtures_per_k=10,
             random_state=None,
             n_jobs=1,
             prior_mixing_coefficients=None,
             prior_emission_probabilities=None,
             **fit_kwargs):

    data = MultiDatasetMixtureModel.collapse_dataset(data)

    if n_jobs > 1:
        pool = multiprocessing.Pool(processes=n_jobs,
                                    initializer=_initializer,
                                    initargs=(data, random_state, mixtures_per_k, fit_kwargs,
                                              prior_mixing_coefficients,
                                              prior_emission_probabilities)
                                    )

        try:
            results = pool.map(_globalised_map_function, k_range_to_search)
        finally:
            pool.close()
    else:
        # Do not spawn an extra process
        results = map(lambda kx: _map_function(kx, mixtures_per_k, data,
                                               fit_kwargs,
                                               random_state,
                                               prior_mixing_coefficients,
                                               prior_emission_probabilities),
                      k_range_to_search)

    results_df = {}
    results_mixtures = {}

    for k, (result, mixture) in zip(k_range_to_search, results):
        results_df[k] = result
        results_mixtures[k] = mixture

    results_df = pd.DataFrame(results_df).T
    results_df.columns.name = 'n_components'

    return results_df, results_mixtures
