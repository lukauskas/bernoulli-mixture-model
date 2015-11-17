from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def bernoulli_prob_for_observations(np.ndarray[np.float_t, ndim=1] p,
                                    np.ndarray[np.uint8_t, cast=True, ndim=2] observations):
    # We are doing
    # emissions = np.power(p, observations) * \
    #             np.power(1 - p, 1 - observations)
    # but in a more efficient way:
    cdef int n
    cdef int d

    cdef n_max = observations.shape[0]
    cdef d_max = observations.shape[1]

    cdef np.float_t row_ans

    cdef np.uint8_t obs

    answer = np.empty(n_max)

    for n in range(n_max):

        row_ans = 1.0

        for d in range(d_max):
            obs = observations[n, d]
            if obs == 1:
                row_ans *= p[d]
            else:
                row_ans *= 1 - p[d]

        answer[n] = row_ans

    return answer