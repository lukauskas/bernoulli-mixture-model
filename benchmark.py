from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

import bernoullimix

def main(max_iter):

    RANDOM_STATE = 12345

    np.random.seed(RANDOM_STATE)

    K = 10
    D = 21
    N = 100000

    mixture_coefficients = np.arange(K, dtype=float) + 1
    mixture_coefficients /= mixture_coefficients.sum()

    emissions = np.random.rand(K, D)

    generator = bernoullimix.BernoulliMixture(K, D, mixture_coefficients, emissions)
    sample, __ = generator.sample(N, random_state=RANDOM_STATE)

    random_bernoulli = next(bernoullimix.random_mixture_generator(K, sample, random_state=RANDOM_STATE))

    print('Fitting for {} iterations'.format(max_iter))
    log_likelihood, converged = random_bernoulli.fit(sample, iteration_limit=max_iter)

    print('Log Likelihood: ', log_likelihood)
    print('converged' if converged else 'did not converge')


if __name__ == '__main__':
    import sys

    try:
        max_iter = int(sys.argv[1])
    except IndexError:
        max_iter = 100

    main(max_iter)