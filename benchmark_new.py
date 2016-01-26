from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import pandas as pd

from bernoullimix.mixture import MultiDatasetMixtureModel

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

    D=len(binary_digits.columns)

    up_missing = binary_digits.iloc[:len(binary_digits)//4].copy()
    bottom_missing = binary_digits.iloc[len(binary_digits)//4:len(binary_digits)//2].copy()
    even_missing = binary_digits.iloc[len(binary_digits)//2:].copy()
    up_missing.iloc[:, :D//2] = None
    bottom_missing.iloc[:, D//2:] = None
    even_missing.iloc[:, np.arange(0, D, 2)] = None

    up_missing['dataset_id'] = 'up_missing'
    bottom_missing['dataset_id'] = 'bottom_missing'
    even_missing['dataset_id'] = 'even_missing'

    training_data = pd.concat((up_missing, bottom_missing, even_missing))

    training_data['weight'] = 1

    return training_data


def get_init_params(data, random_state):

    data_cols = data.columns - ['dataset_id', 'weight']

    random = np.random.RandomState(random_state)

    mu = data.dataset_id.value_counts() / len(data)
    mu.name = 'mu'

    K = 10
    D = len(data_cols)
    C = len(mu)

    pi = pd.DataFrame(random.rand(C, K), index=mu.index, columns=["K{}".format(k) for k in range(K)])
    pi = pi.divide(pi.sum(axis=1), axis=0)

    p = pd.DataFrame(random.rand(K, D), index=pi.columns, columns=data_cols)

    return mu, pi, p

def main(max_iter):

    RANDOM_STATE = 125

    np.random.seed(RANDOM_STATE)
    data = load_data(random_state=RANDOM_STATE)

    mu, pi, p = get_init_params(data, RANDOM_STATE)

    print('MU:')
    print(mu)
    print('PI:')
    print(pi.sum(axis='columns'))
    print('P:')
    print(p)

    model = MultiDatasetMixtureModel(mu, pi, p)

    print(model.fit(data, n_iter=max_iter))


if __name__ == '__main__':
    import sys

    try:
        max_iter = int(sys.argv[1])
    except IndexError:
        max_iter = 50

    main(max_iter)