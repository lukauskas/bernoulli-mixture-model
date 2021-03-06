from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd


from bernoullimix.mixture import MultiDatasetMixtureModel, DATASET_ID_COLUMN, WEIGHT_COLUMN


def _adjust_probabilities(unadjusted_array, epsilon, domain=(0, 1)):
    """
    Adjusts the values in an array that are all in the specified `domain`
    to be within the domain [`epsilon`, `1-epsilon`].

    :param unadjusted_array:
    :param epsilon:
    :param domain:
    :return:
    """

    if domain[1] <= domain[0]:
        raise ValueError('Incorrect domain specified, expecting domain[0] < domain[1]')

    domain_width = domain[1] - domain[0]
    projected_domain_width = 1 - 2 * epsilon

    def _adjust(x):
        if x < domain[0] or x > domain[1]:
            raise ValueError('value {!r} not within the domain [{}, {}]'.format(x, domain[0], domain[1]))

        return ((x - domain[0]) / domain_width) * projected_domain_width + epsilon

    adjust = np.vectorize(_adjust)

    return adjust(unadjusted_array)


def _expected_domain(range_a, range_b, alpha):
    """
    Computes the expected domain for the weighted of two ranges as follows:

        `alpha * A + (1-alpha) B`
    where  `range_a` and `range_b` are the possible ranges for components A and B

    :param range_a: range for k A
    :param range_b: range for k B
    :param alpha: mixing parameter
    :return: a tuple of min and max values for the sum domain.
    """
    range_a = np.asarray(range_a)
    range_b = np.asarray(range_b)

    if len(range_a) != 2 or len(range_b) != 2:
        raise ValueError('Ranges provided should be of length 2')

    if range_a[0] > range_a[1] or range_b[0] > range_b[1]:
        raise ValueError('Invalid ranges provided. First value should be lower than the second')

    if not (0 <= alpha <= 1):
        raise ValueError('Expecting alpha to be between 0 and 1, got: {!r}'.format(alpha))

    return tuple(range_a * alpha + range_b * (1-alpha))

def _random_numbers_within_domain(random, domain, shape):
    """
    Generates random numbers within the specified domain
    :param random:
    :type random: np.random.RandomState
    :param domain: domain to compute numbers for
    :param shape: shape of array to produce
    :return:
    """

    # Generates numbers between 0 and 1
    random_numbers = random.rand(*shape)

    domain_width = domain[1] - domain[0]

    # Shift the random numbers to be within correct domain
    random_numbers *= domain_width
    random_numbers += domain[0]

    return random_numbers


def _random_rows_from_dataset(dataset, n_rows, random):
    random_rows = random.randint(len(dataset), size=n_rows)
    random_row_emissions = dataset.iloc[random_rows].copy()

    for column_name in random_row_emissions.columns:
        nulls = random_row_emissions[column_name].isnull()
        n_nulls = nulls.sum()

        if not n_nulls:
            continue


        random_binary_ints = random.randint(2, size=n_nulls)

        random_bools = np.array(random_binary_ints, dtype=bool)
        random_row_emissions.loc[nulls, column_name] = random_bools

    assert random_row_emissions.isnull().sum().sum() == 0

    return random_row_emissions.astype(bool)

def random_mixture_generator(number_of_components,
                             dataset,
                             random_state=None,
                             epsilon=0.005,
                             alpha=0.75,
                             prior_mixing_coefficients=None,
                             prior_emission_probabilities=None):
    """
    Returns a generator for `MultiDatasetBernoulli` initialiser.
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
    :param prior_mixing_coefficients: priors for MAP estimation of mixing coefficients
    :param prior_emission_probabilities: priors for MAP estimation of emission probabilities
    :return:
    """

    dataset = pd.DataFrame(dataset)

    assert DATASET_ID_COLUMN in dataset.columns
    assert WEIGHT_COLUMN in dataset.columns

    data = dataset[dataset.columns.difference([DATASET_ID_COLUMN, WEIGHT_COLUMN])]
    dataset_id_counts = dataset[DATASET_ID_COLUMN].value_counts()

    random = np.random.RandomState(random_state)
    components_index = ['K{}'.format(k) for k in range(number_of_components)]
    mixing_coefficients = pd.DataFrame([np.repeat(1/number_of_components, number_of_components)],
                                       index=dataset_id_counts.index,
                                       columns=components_index)

    random_domain = (0, 1)
    dataset_domain = (0, 1)

    expected_domain = _expected_domain(random_domain,
                                       dataset_domain,
                                       alpha=alpha)
    while True:

        N, D = data.shape

        random_emissions = _random_numbers_within_domain(random, random_domain,
                                                         (number_of_components, D))

        random_row_emissions = _random_rows_from_dataset(data, n_rows=number_of_components,
                                                         random=random)

        random_row_emissions = np.asarray(random_row_emissions, dtype=bool)

        emissions = alpha * random_emissions + (1-alpha) * random_row_emissions
        emissions = _adjust_probabilities(emissions, epsilon, domain=expected_domain)

        emissions = pd.DataFrame(emissions,
                                 index=components_index,
                                 columns=data.columns)

        yield MultiDatasetMixtureModel(mixing_coefficients, emissions,
                                       prior_emission_probabilities=prior_emission_probabilities,
                                       prior_mixing_coefficients=prior_mixing_coefficients)


