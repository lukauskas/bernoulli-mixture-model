from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import pandas as pd

from bernoullimix.mixture import MultiDatasetMixtureModel
from bernoullimix.random_initialisation import random_mixture_generator

MASKS = {'top': np.array([[True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [False, False, False, False, False, False, False, False],
                          [False, False, False, False, False, False, False, False],
                          [False, False, False, False, False, False, False, False],
                          [False, False, False, False, False, False, False, False]]
                         ),
         'bottom': np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]]
                            ),
         'stripes': np.array([[True, False, True, False, True, False, True, False],
                              [True, False, True, False, True, False, True, False],
                              [True, False, True, False, True, False, True, False],
                              [True, False, True, False, True, False, True, False],
                              [True, False, True, False, True, False, True, False],
                              [True, False, True, False, True, False, True, False],
                              [True, False, True, False, True, False, True, False],
                              [True, False, True, False, True, False, True, False], ]),
         }
RESHAPED_MASKS = {key: np.reshape(mask, -1) for key, mask in MASKS.items()}

MASK_PROPORTIONS = pd.Series([0.2, 0.3, 0.5], index=['top', 'stripes', 'bottom'])
N_REPEATS = 3

def load_digits(random_state):

    import sklearn.datasets
    digits_dataset = sklearn.datasets.load_digits()
    digits = pd.DataFrame(digits_dataset.data)
    labels = pd.Series(digits_dataset.target, index=digits.index)

    THRESHOLD = np.mean(digits.values.reshape(-1))
    binary_digits = digits >= THRESHOLD

    from sklearn.utils import shuffle
    binary_digits = shuffle(binary_digits, random_state=random_state)

    dataset = binary_digits.copy()
    dataset['dataset_id'] = None
    sum_proportions = 0
    for mask, proportion in MASK_PROPORTIONS.iteritems():
        dataset['dataset_id'].iloc[int(sum_proportions * len(dataset)):int(
            (sum_proportions + proportion) * len(dataset))] = mask
        sum_proportions += proportion

    for name, mask in RESHAPED_MASKS.items():
        dataset.loc[dataset['dataset_id'] == name, ~mask] = np.nan

    dataset['weight'] = 1

    return dataset


def load_random(rows, columns, random_state):
    random = np.random.RandomState(random_state)
    data = random.binomial(1, 0.5, size=rows*columns).reshape(rows, columns)

    data = pd.DataFrame(data)
    data['dataset_id'] = 'test'
    data['weight'] = 1

    return data

def main(max_iter, K):

    RANDOM_STATE = 125

    np.random.seed(RANDOM_STATE)
    # data = load_digits(random_state=RANDOM_STATE)
    data = load_random(10000, 20, random_state=RANDOM_STATE)
    print('Dataset shape: {:,}x{:,}'.format(data.shape[0], data.shape[1]))

    for i in range(N_REPEATS):
        model = next(random_mixture_generator(K, data,
                                              random_state=RANDOM_STATE,
                                              prior_mixing_coefficients=2,
                                              prior_emission_probabilities=(2, 2),
                                              ))

        # print('MU:')
        # print(model.dataset_priors)
        # print('PI:')
        # print(model.mixing_coefficients)
        # print('P:')
        # print(model.emission_probabilities.min().min(), model.emission_probabilities.max().max())

        print(model.fit(data, n_iter=max_iter, eps=1e-2))


if __name__ == '__main__':
    import sys
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    try:
        max_iter = int(sys.argv[1])
    except IndexError:
        max_iter = 1000

    try:
        K = int(sys.argv[2])
    except IndexError:
        K = 10

    main(max_iter, K)
