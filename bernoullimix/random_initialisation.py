from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from bernoullimix.mixture import BernoulliMixture

def random_mixture_generator(number_of_components,
                             dataset,
                             random_state=None,
                             epsilon=0.005,
                             alpha=0.75):
    """
    Returns a generator for `BernoulliMixture` initialiser.
    The mixing coefficients are always chosen uniform.

    Emission probabilities are generated from

        alpha * rand_component + (1-alpha) random_row_from_data

    As described in "EM initialisation for Bernoulli Mixture learning" by A. Juan, et al.

    Probabilities are also smoothed to be within range [epsilon, 1-epsilon].

    :param number_of_components: number of components to generate samples for
    :param dataset: dataset to use for initialisation
    :param random_state: random seed
    :param epsilon: probabilities will be adjusted to be within range [epsilon, 1-epsilon]
    :param alpha: mixing coefficient for random coefficient and random row from data
    :return:
    """

    random = np.random.RandomState(random_state)

    mixing_coefficients = np.repeat(1/number_of_components, number_of_components)

    fix_range = np.vectorize(lambda x: epsilon + (x/( (1 - epsilon) - epsilon)))

    while True:

        N, D = dataset.shape

        random_emissions = random.rand(number_of_components, D)

        random_rows = random.randint(N, size=number_of_components)

        random_row_emissions = dataset[random_rows, :]

        emissions = alpha * random_emissions + (1-alpha) * random_row_emissions
        emissions = fix_range(emissions)

        yield BernoulliMixture(number_of_components, D,
                               mixing_coefficients, emissions)


