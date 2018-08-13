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
TRAIN_SIZE = 0.7

def load_data(random_state):

    import sklearn.datasets
    digits_dataset = sklearn.datasets.load_digits()
    digits = pd.DataFrame(digits_dataset.data)
    labels = pd.Series(digits_dataset.target, index=digits.index)

    THRESHOLD = np.mean(digits.values.reshape(-1))
    binary_digits = digits >= THRESHOLD

    from sklearn.utils import shuffle
    binary_digits = shuffle(binary_digits, random_state=random_state)
    labels = labels.loc[binary_digits.index]

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

    from sklearn.model_selection import train_test_split
    train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels,
                                                                              train_size=TRAIN_SIZE,
                                                                              random_state=random_state)

    DATASETS = {'split': {'train': train_dataset, 'test': test_dataset}}
    TRUE_STATES = {'train': train_labels, 'test': test_labels}

    DATASETS['unified'] = {}
    for type_, dataset in DATASETS['split'].items():
        unified_dataset = dataset.copy()
        unified_dataset['dataset_id'] = 'unified'

        DATASETS['unified'][type_] = unified_dataset

    return DATASETS

def main(max_iter, K):

    RANDOM_STATE = 125

    np.random.seed(RANDOM_STATE)
    data = load_data(random_state=RANDOM_STATE)
    data = data['split']['train']
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
