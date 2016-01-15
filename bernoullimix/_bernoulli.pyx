from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
cimport numpy as np
cimport cython



from libc.math cimport log

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bernoulli_prob_for_observations_with_mask(np.float_t[:] p,
                                                np.ndarray[np.uint8_t, cast=True, ndim=2] observations,
                                                np.ndarray[np.uint8_t, cast=True, ndim=2] observed_mask):
    # We are doing
    # emissions = np.power(p, observations) * \
    #             np.power(1 - p, 1 - observations)
    # but in a more efficient way:
    cdef int n
    cdef int d

    cdef int n_max = observations.shape[0]
    cdef int d_max = observations.shape[1]

    assert observed_mask.shape[0] == observations.shape[0]
    assert observed_mask.shape[1] == observations.shape[1]

    cdef np.float_t row_ans

    cdef np.uint8_t obs

    cdef np.ndarray[np.float_t, ndim=1] answer = np.empty(n_max, dtype=np.float)

    for n in range(n_max):

        row_ans = 1.0

        for d in range(d_max):
            if observed_mask[n, d] == 1:
                obs = observations[n, d]
                if obs == 1:
                    row_ans *= p[d]
                else:
                    row_ans *= 1.0 - p[d]

        answer[n] = row_ans

    return answer

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef probability_z_o_given_theta_c(np.ndarray[np.uint8_t, cast=True, ndim=2] observations,
                                    np.ndarray[np.uint8_t, cast=True, ndim=2] observed_mask,
                                    np.ndarray[np.float_t, ndim=2] emission_probabilities,
                                    np.ndarray[np.float_t, ndim=1] mixing_coefficients):

    cdef int N = observations.shape[0]
    cdef int K = mixing_coefficients.shape[0]

    cdef np.ndarray[np.float_t, ndim=2] answer = np.empty((N, K), dtype=np.float, order='F')

    cdef int component
    cdef np.float_t[:] component_emission_probs

    for component in range(K):
        component_emission_probs = emission_probabilities[component]

        answer[:, component] = mixing_coefficients[component] * \
                               bernoulli_prob_for_observations_with_mask(component_emission_probs,
                                                                         observations,
                                                                         observed_mask)

    return answer

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def impute_missing_data_c(
        np.ndarray[np.uint8_t, cast=True, ndim=2] observations,
        np.ndarray[np.uint8_t, cast=True, ndim=2] observed_mask,
        np.ndarray[np.float_t, ndim=2] emission_probabilities,
        np.ndarray[np.float_t, ndim=1] mixing_coefficients):

    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]
    cdef int K = mixing_coefficients.shape[0]

    cdef np.ndarray[np.float_t, ndim=2] answer

    answer = np.empty((N, D), dtype=np.float)

    cdef np.float_t[:, :] S = probability_z_o_given_theta_c(observations,
                                                            observed_mask,
                                                            emission_probabilities,
                                                            mixing_coefficients)

    cdef int k
    cdef np.float_t p;
    cdef np.float_t num;
    cdef np.float_t denom;

    for n in range(N):
        for d in range(D):
            if observed_mask[n,d]:
                answer[n, d] = observations[n, d]
            else:
                num = 0
                denom = 0
                for k in range(K):
                    p = emission_probabilities[k, d]
                    num += p * S[n, k]
                    denom += S[n, k]
                answer[n, d] = num/denom

    return answer

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.float_t _log_likelihood_from_z_o_joint(np.float_t[:,:] z_o_joint,
                                                np.int_t[:] weights):


    cdef int N = z_o_joint.shape[0]
    cdef int K = z_o_joint.shape[1]

    cdef np.float_t row_log_likelihood
    cdef np.float_t ans = 0

    cdef int n;
    cdef int k;

    for n in range(N):
        row_log_likelihood = 0
        for k in range(K):
            row_log_likelihood += z_o_joint[n, k]

        row_log_likelihood = log(row_log_likelihood)
        row_log_likelihood *= weights[n]

        ans += row_log_likelihood

    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float_t, ndim=2] _posterior_probability_of_class_given_support(np.float_t[:,:] support):

    cdef int N = support.shape[0]
    cdef int K = support.shape[1]

    cdef np.ndarray[np.float_t, ndim=2] ans = np.empty((N,K), dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=1] support_sums = np.zeros(N, dtype=np.float)

    cdef int n;

    for k in range(K):
        for n in range(N):
            support_sums[n] += support[n, k]

    for k in range(K):
        for n in range(N):
            ans[n, k] = support[n, k] /  support_sums[n]

    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
def _m_step(np.ndarray[np.uint8_t, cast=True, ndim=2] unique_dataset,
            np.ndarray[np.uint8_t, cast=True, ndim=2] unique_mask,
            np.float_t[:,:] unique_zstar,
            np.int_t[:] weights,
            np.ndarray[np.float_t, ndim=2] old_emission_probabilities
            ):

    cdef int N = unique_zstar.shape[0]
    cdef int K = unique_zstar.shape[1]

    cdef int D = unique_dataset.shape[1]

    assert unique_mask.shape[0] == N
    assert unique_mask.shape[1] == D

    assert old_emission_probabilities.shape[0] == K
    assert old_emission_probabilities.shape[1] == D

    cdef int n;
    cdef int k;
    cdef int d;

    cdef np.float_t zstar_times_weight;

    cdef np.ndarray[np.float_t, ndim=2] e = np.zeros((K, D), dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=1] c = np.zeros(K, dtype=np.float)

    for k in range(K):
        for n in range(N):

            zstar_times_weight = unique_zstar[n, k] * weights[n]
            c[k] += zstar_times_weight

            for d in range(D):
                if unique_mask[n, d]:
                    e[k, d] += unique_dataset[n, d] * zstar_times_weight
                else:
                    e[k, d] += old_emission_probabilities[k, d] * zstar_times_weight

        for d in range(D):
            e[k, d] /= c[k]

    cdef np.float_t sum_of_c = 0
    for k in range(K):
        sum_of_c += c[k]

    for k in range(K):
        c[k] /= sum_of_c

    return c, e

@cython.boundscheck(False)
@cython.wraparound(False)
def _em(np.ndarray[np.uint8_t, cast=True, ndim=2] unique_dataset,
        np.ndarray[np.int_t, ndim=1] counts,
        np.ndarray[np.uint8_t, cast=True, ndim=2] observed_mask,
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
    cdef np.float_t[:,:] previous_unique_support
    cdef np.float_t[:,:] current_unique_support

    cdef np.float_t previous_log_likelihood
    cdef np.float_t current_log_likelihood

    cdef np.float_t[:,:] unique_zstar

    previous_unique_support = probability_z_o_given_theta_c(unique_dataset, observed_mask,
                                                            emission_probabilities,
                                                            mixing_coefficients)

    previous_log_likelihood = _log_likelihood_from_z_o_joint(previous_unique_support, counts)

    if trace_likelihood:
        likelihood_trace.append(previous_log_likelihood)


    while iteration_limit < 0 or iterations_done < iteration_limit:

        unique_zstar = _posterior_probability_of_class_given_support(previous_unique_support)
        mixing_coefficients, emission_probabilities = _m_step(unique_dataset,
                                                              observed_mask,
                                                              unique_zstar, counts,
                                                              emission_probabilities
                                                              )

        current_unique_support = probability_z_o_given_theta_c(unique_dataset, observed_mask,
                                                               emission_probabilities,
                                                               mixing_coefficients)
        current_log_likelihood = _log_likelihood_from_z_o_joint(current_unique_support, counts)

        iterations_done += 1

        if trace_likelihood:
            likelihood_trace.append(current_log_likelihood)

        if current_log_likelihood - previous_log_likelihood < convergence_threshold:
            try:
                assert current_log_likelihood - previous_log_likelihood >= 0, \
                    'Likelihood decreased by {} ' \
                    'in iteration {}'.format(previous_log_likelihood-current_log_likelihood,
                                             iterations_done)
            except AssertionError:
                raise
            converged = 1
            break

        previous_log_likelihood = current_log_likelihood
        previous_unique_support = current_unique_support



    if trace_likelihood:
        likelihood_trace = np.array(likelihood_trace)

    return mixing_coefficients, emission_probabilities, \
           converged, current_log_likelihood, iterations_done, likelihood_trace