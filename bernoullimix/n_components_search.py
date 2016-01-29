from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import itertools
from bernoullimix import MultiDatasetMixtureModel
from bernoullimix.mixture import WEIGHT_COLUMN
from bernoullimix.random_initialisation import random_mixture_generator
import pandas as pd


def search_k(k_range_to_search, data, components_per_k=10, random_state=None, **fit_kwargs):

    data = MultiDatasetMixtureModel.collapse_dataset(data)
    sum_of_weights = data[WEIGHT_COLUMN].sum()

    best_mixtures = {}
    best_results = {}

    for k in k_range_to_search:
        mixtures_k = list(itertools.islice(random_mixture_generator(k, data, random_state=random_state),
                                           components_per_k))

        results = map(lambda x: x.fit(data, **fit_kwargs), mixtures_k)

        best_result = None
        best_mixture = None

        for result, mixture in zip(results, mixtures_k):
            result = pd.Series(result, index=['converged', 'n_iterations', 'log_likelihood'])
            if best_result is None or result['log_likelihood'] > best_result['log_likelihood']:
                best_result = result
                best_mixture = mixture

        best_mixtures[k] = best_mixture
        best_results[k] = best_result

    best_results = pd.DataFrame(best_results).T

    bics = {}
    for k in k_range_to_search:
        mixture = best_mixtures[k]
        bic = mixture.BIC(best_results.loc[k, 'log_likelihood'], sum_of_weights)

        bics[k] = bic

    bics = pd.Series(bics)
    best_results['BIC'] = bics

    return best_mixtures, best_results
