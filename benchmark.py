from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import bernoullimix



def main(max_iter):

    RANDOM_STATE = 12345

    np.random.seed(RANDOM_STATE)

    K = 100
    D = 21
    N = 100000

    mixture_coefficients = np.arange(K, dtype=float)
    mixture_coefficients /= mixture_coefficients.sum()

    emissions = np.random.rand(K, D)

    generator = bernoullimix.BernoulliMixture(K, D, mixture_coefficients, emissions)
    sample, __ = generator.sample(N, random_state=RANDOM_STATE)

    random_bernoulli = next(bernoullimix.random_mixture_generator(K, sample,
                                                                  random_state=RANDOM_STATE))

    starting_log_likelihood = random_bernoulli.log_likelihood(sample)

    starting_mc = random_bernoulli.mixing_coefficients
    starting_ep = random_bernoulli.emission_probabilities

    print('Fitting for {} iterations'.format(max_iter))
    log_likelihood, converged = random_bernoulli.fit(sample, iteration_limit=max_iter)

    print('Log Likelihood: ', log_likelihood)
    print(converged)

    final_log_likelihood = random_bernoulli.log_likelihood(sample)

    print('Likelihood increased by: {}'.format(final_log_likelihood - starting_log_likelihood))
    assert final_log_likelihood > starting_log_likelihood

    final_mc = random_bernoulli.mixing_coefficients
    final_ep = random_bernoulli.emission_probabilities

    print('Observed mixing cooefficients:\n{}'.format(final_mc))


if __name__ == '__main__':
    import sys

    try:
        max_iter = int(sys.argv[1])
    except IndexError:
        max_iter = 50

    main(max_iter)