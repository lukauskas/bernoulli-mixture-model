# cython: profile=True
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

@cython.boundscheck(False)
@cython.wraparound(False)
def maximise_emissions(np.ndarray[np.uint8_t, cast=True, ndim=2] unique_dataset,
                       np.ndarray[np.float_t, ndim=2] unique_zstar,
                       np.ndarray[np.int64_t, ndim=1] weights):
    cdef int N = unique_zstar.shape[0]
    cdef int K = unique_zstar.shape[1]

    cdef int D = unique_dataset.shape[1]

    cdef int n;
    cdef int k;
    cdef int d;

    cdef np.float_t v_kd;

    cdef np.ndarray[np.float_t, ndim=2] v = np.zeros((K, D), dtype=np.float)

    for k in range(K):
        for n in range(N):
            for d in range(D):
                v[k, d] += unique_dataset[n, d] * unique_zstar[n, k] * weights[n]

    return v

@cython.inline
cpdef _log_likelihood_from_support(np.ndarray[np.float_t, ndim=2] support,
                                   np.ndarray[np.int64_t, ndim=1] weights):

    return np.sum(np.log(np.sum(support, axis=1)) * weights)


@cython.inline
cpdef _posterior_probability_of_class_given_support(support):
    return (support.T / np.sum(support, axis=1)).T

cpdef _m_step(unique_zstar, unique_dataset, weights):
    u = np.sum(unique_zstar.T * weights, axis=1)
    sum_of_weights = np.sum(u)

    N, K = unique_zstar.shape

    vs = maximise_emissions(unique_dataset, unique_zstar, weights)

    for k in range(K):
        vs[k] /= u[k]

    return u / sum_of_weights, vs

def _em(unique_dataset, counts,
        mixing_coefficients, emission_probabilities,
        iteration_limit, convergence_threshold, trace_likelihood):

    iterations_done = 0
    previous_log_likelihood, current_log_likelihood = None, None
    if trace_likelihood:
        likelihood_trace = []
    else:
        likelihood_trace = None

    converged = False
    while iteration_limit is None or iterations_done < iteration_limit:

        unique_support = observation_emission_support_c(unique_dataset,
                                                        emission_probabilities,
                                                        mixing_coefficients)

        current_log_likelihood = _log_likelihood_from_support(unique_support, counts)
        if previous_log_likelihood is not None \
                and np.isclose(current_log_likelihood, previous_log_likelihood,
                               rtol=0, atol=convergence_threshold):
            converged = True
            break

        unique_z_star = _posterior_probability_of_class_given_support(unique_support)

        pi, e = _m_step(unique_z_star, unique_dataset, counts)

        mixing_coefficients = pi
        emission_probabilities = e

        if trace_likelihood:
            likelihood_trace.append(current_log_likelihood)

        previous_log_likelihood = current_log_likelihood

        iterations_done += 1

    if trace_likelihood:
        likelihood_trace = np.array(likelihood_trace)

    return converged, current_log_likelihood, iterations_done, likelihood_trace