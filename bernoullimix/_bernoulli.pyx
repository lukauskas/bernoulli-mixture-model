from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bernoulli_prob_for_observations(np.ndarray[np.float_t, ndim=1] p,
                                      np.ndarray[np.uint8_t, cast=True, ndim=2] observations):
    # We are doing
    # emissions = np.power(p, observations) * \
    #             np.power(1 - p, 1 - observations)
    # but in a more efficient way:
    cdef int n
    cdef int d

    cdef int n_max = observations.shape[0]
    cdef int d_max = observations.shape[1]

    cdef np.float_t row_ans

    cdef np.uint8_t obs

    cdef np.ndarray[np.float_t, ndim=1] answer

    answer = np.empty(n_max, dtype=np.float)

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

@cython.boundscheck(False)
@cython.wraparound(False)
def observation_emission_support_c(
        np.ndarray[np.uint8_t, cast=True, ndim=2] observations,
        np.ndarray[np.float_t, ndim=2] emission_probabilities,
        np.ndarray[np.float_t, ndim=1] mixing_coefficients):

    cdef int N = observations.shape[0]
    cdef int K = mixing_coefficients.shape[0]

    cdef np.ndarray[np.float_t, ndim=2] answer

    answer = np.empty((N, K), dtype=np.float, order='F')

    cdef int component

    for component in range(K):
        component_emission_probs = emission_probabilities[component]

        answer[:, component] = mixing_coefficients[component] * \
                               bernoulli_prob_for_observations(component_emission_probs,
                                                               observations)

    return answer

@cython.inline
cpdef _log_likelihood_from_support(np.ndarray[np.float_t, ndim=2] support,
                                   np.ndarray[np.int64_t, ndim=1] weights):

    return np.sum(np.log(np.sum(support, axis=1)) * weights)


@cython.inline
cpdef _posterior_probability_of_class_given_support(np.ndarray[np.float_t, ndim=2] support):
    return (support.T / np.sum(support, axis=1)).T

@cython.boundscheck(False)
@cython.wraparound(False)
def _m_step(np.ndarray[np.uint8_t, cast=True, ndim=2] unique_dataset,
            np.ndarray[np.float_t, ndim=2] unique_zstar,
            np.ndarray[np.int64_t, ndim=1] weights):
    cdef int N = unique_zstar.shape[0]
    cdef int K = unique_zstar.shape[1]

    cdef int D = unique_dataset.shape[1]

    cdef int n;
    cdef int k;
    cdef int d;

    cdef np.float_t zstar_times_weight;

    cdef np.ndarray[np.float_t, ndim=2] v = np.zeros((K, D), dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=1] u = np.zeros(K, dtype=np.float)

    for k in range(K):
        for n in range(N):

            zstar_times_weight = unique_zstar[n, k] * weights[n]
            u[k] += zstar_times_weight

            for d in range(D):
                v[k, d] += unique_dataset[n, d] * zstar_times_weight

        v[k] = v[k] / u[k]

    cdef np.float_t sum_of_u = 0
    for k in range(K):
        sum_of_u += u[k]

    for k in range(K):
        u[k] /= sum_of_u

    return u, v

@cython.boundscheck(False)
@cython.wraparound(False)
def _em(np.ndarray[np.uint8_t, cast=True, ndim=2] unique_dataset,
        np.ndarray[np.int64_t, ndim=1] counts,
        np.ndarray[np.float_t, ndim=1] mixing_coefficients,
        np.ndarray[np.float_t, ndim=2] emission_probabilities,
        int iteration_limit,
        np.float_t convergence_threshold,
        int trace_likelihood):

    cdef int iterations_done = 0

    if trace_likelihood:
        likelihood_trace = []
    else:
        likelihood_trace = None

    cdef int converged = 0
    cdef np.ndarray[np.float_t, ndim=2] previous_unique_support
    cdef np.ndarray[np.float_t, ndim=2] current_unique_support

    cdef np.float_t previous_log_likelihood
    cdef np.float_t current_log_likelihood

    cdef np.ndarray[np.float_t, ndim=2] unique_zstar

    previous_unique_support = observation_emission_support_c(unique_dataset,
                                                             emission_probabilities,
                                                             mixing_coefficients)

    previous_log_likelihood = _log_likelihood_from_support(previous_unique_support, counts)

    if trace_likelihood:
        likelihood_trace.append(previous_log_likelihood)

    while iteration_limit < 0 or iterations_done < iteration_limit:

        unique_zstar = _posterior_probability_of_class_given_support(previous_unique_support)
        mixing_coefficients, emission_probabilities = _m_step(unique_dataset, unique_zstar, counts)

        current_unique_support = observation_emission_support_c(unique_dataset,
                                                                emission_probabilities,
                                                                mixing_coefficients)
        current_log_likelihood = _log_likelihood_from_support(current_unique_support, counts)

        iterations_done += 1

        if trace_likelihood:
            likelihood_trace.append(current_log_likelihood)

        if current_log_likelihood - previous_log_likelihood < convergence_threshold:
            assert current_log_likelihood - previous_log_likelihood >= 0
            converged = 1
            break

        previous_log_likelihood = current_log_likelihood
        previous_unique_support = current_unique_support



    if trace_likelihood:
        likelihood_trace = np.array(likelihood_trace)

    return mixing_coefficients, emission_probabilities, \
           converged, current_log_likelihood, iterations_done, likelihood_trace