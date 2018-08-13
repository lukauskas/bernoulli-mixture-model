import numpy as np
cimport numpy as np
cimport cython

from cpython cimport array
import array
from libc.math cimport log


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unnormalised_pi_weights(np.float64_t[:] pi_prior,
                              np.float64_t[:,:] pi):


    cdef int n_datasets = pi.shape[0]
    cdef int K = pi.shape[1]

    cdef int i,j
    cdef np.float64_t pi_weights = 0
    with nogil:
        for i in range(n_datasets):
            for j in range(K):
                pi_weights += (pi_prior[j]-1) * log(pi[i, j])

    return pi_weights


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unnormalised_p_weights(np.float64_t[:,:] p_prior,
                             np.float64_t[:,:] p):


    cdef int K = p.shape[0]
    cdef int D = p.shape[1]

    cdef int i,j

    cdef np.float64_t p_weights = 0
    with nogil:
        for i in range(K):
            for j in range(D):

                p_weights += log(p[i,j]) * (p_prior[j, 0] - 1)
                p_weights += log(1 - p[i,j]) * (p_prior[j, 1] - 1)

    return p_weights

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef p_update(np.uint8_t[:,:] observations,
               np.uint8_t[:,:] not_null_mask,
               np.float64_t[::1,:] zstar_times_weight,
               np.float64_t[:,:] old_p,
               np.float64_t[:,:] p_priors):

    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]
    cdef int K = old_p.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] new_p = np.zeros((K, D), dtype=np.float64)

    cdef np.float64_t weight_sum;
    cdef np.float64_t zw;

    cdef int k, n, d

    with nogil:
        for k in range(K):
            weight_sum = 0.0
            for n in range(N):
                zw = zstar_times_weight[n, k]
                weight_sum += zw

                for d in range(D):
                    if not_null_mask[n, d]:
                        new_p[k, d] += observations[n, d] * zw
                    else:
                        new_p[k, d] += old_p[k, d] * zw

            for d in range(D):
                new_p[k, d] += p_priors[d, 0] - 1 # plus alpha_d - 1
                new_p[k, d] /= weight_sum + p_priors[d, 0] + p_priors[d, 1] - 2 # + alpha_d + beta_d - 2

                if new_p[k, d] > 1.0:
                   # assert new_p[k, d] - 1.0 < 1e-10
                    new_p[k, d] = 1.0
    return new_p


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef log_support_c(np.uint8_t[:,:] observations,
                    np.uint8_t[:,:] not_null_mask,
                    np.int_t[:] dataset_ids_as_ilocs,
                    np.float64_t[:,:] pi,
                    np.float64_t[:,:] p):
    """
    Computes log of unnormalised `z*`.
    That is, for all rows i and components k, it computes

    pi_k,c * product( p_{k,d}^{x_d^(i)} (1-p_{k,d})^(1 - x_d^(i))) for all d in observed variables for i

    where c is the index of dataset that generated row i

    the returned value is the logarithm of this, i.e.:

    pi_k,c + sum( (x_d^i) log p_{k,d} + (1-x_d^i) log(1 - p_{k_d}) ) for all d

    :param observations:
    :param not_null_mask:
    :param dataset_ids_as_ilocs:
    :param pi:
    :param p:
    :return:
    """

    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]
    cdef int K = pi.shape[1]
    cdef int n_datasets = pi.shape[0]
    cdef np.float64_t row_ans

    cdef np.ndarray[np.float64_t, ndim=2] ans = np.empty((N, K), dtype=float, order='F')

    cdef np.float64_t[:, :] log_pi = np.empty((n_datasets, K), dtype=np.float64)
    cdef np.float64_t[:, :] log_p = np.empty((K, D), dtype=np.float64)
    cdef np.float64_t[:, :] log_one_minus_p = np.empty((K, D), dtype=np.float64)

    cdef int k, n, d, i

    with nogil:
        for k in range(K):
            for d in range(D):
                log_p[k, d] = log(p[k, d])
                log_one_minus_p[k, d] = log(1 - p[k,d])

        for i in range(n_datasets):
            for k in range(K):
                log_pi[i, k] = log(pi[i, k])


        for k in range(K):
            for n in range(N):

                row_ans = 0
                for d in range(D):
                    if not_null_mask[n, d]:
                        if observations[n, d]:
                            row_ans += log_p[k, d]
                        else:
                            row_ans += log_one_minus_p[k, d]

                # assert not np.isnan(row_ans)
                ans[n, k] = row_ans + log_pi[dataset_ids_as_ilocs[n], k]


    return ans
