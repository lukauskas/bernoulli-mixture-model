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

def _initializer(data, random_state, n_mixtures_to_search, fit_kwargs):
    global g_data, g_random_state, g_n_mixtures_to_search, g_fit_kwargs

    n_mixtures_to_search = int(n_mixtures_to_search)
    g_n_mixtures_to_search = n_mixtures_to_search

    g_data = data
    g_random_state = random_state
    g_fit_kwargs = fit_kwargs


def _map_function(k):
    global g_data, g_random_state, g_n_mixtures_to_search, g_fit_kwargs

    generator = random_mixture_generator(k, g_data,
                                         random_state=g_random_state)

    mixtures = list(itertools.islice(generator, g_n_mixtures_to_search))

    best_result = None
    best_mixture = None

    for mixture in mixtures:
        result = mixture.fit(g_data, **g_fit_kwargs)
        result = pd.Series(result, index=['converged', 'n_iterations', 'log_likelihood'])
        if best_result is None or result['log_likelihood'] > best_result['log_likelihood']:
            best_result = result
            best_mixture = mixture

    best_result['BIC'] = best_mixture.BIC(best_result['log_likelihood'],
                                          g_data[WEIGHT_COLUMN].sum())

    return best_result, best_mixture


def search_k(k_range_to_search, data, mixtures_per_k=10,
             random_state=None,
             n_jobs=1,
             **fit_kwargs):

    data = MultiDatasetMixtureModel.collapse_dataset(data)

    pool = multiprocessing.Pool(processes=n_jobs,
                                initializer=_initializer,
                                initargs=(data, random_state, mixtures_per_k, fit_kwargs)
                                )

    results = pool.map(_map_function, k_range_to_search)

    results_df = {}
    results_mixtures = {}

    for k, (result, mixture) in zip(k_range_to_search, results):
        results_df[k] = result
        results_mixtures[k] = mixture

    results_df = pd.DataFrame(results_df).T
    results_df.columns.name = 'n_components'

    return results_df, results_mixtures
