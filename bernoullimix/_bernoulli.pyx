from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np


def bernoulli_prob_for_observations(p, observations):
    # We are doing
    # emissions = np.power(p, observations) * \
    #             np.power(1 - p, 1 - observations)
    # but in a more efficient way:
    emissions = np.tile(p, (len(observations), 1))
    emissions[~observations] = 1 - emissions[~observations]
    # and then computing the product
    return np.product(emissions, axis=1)
