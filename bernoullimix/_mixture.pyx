import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef p_update(np.ndarray[np.uint8_t, ndim=2, cast=True] observations,
               np.ndarray[np.uint8_t, ndim=2, cast=True] not_null_mask,
               np.float64_t[::1,:] zstar_times_weight,
               np.float64_t[:,:] old_p,
               np.float64_t[:,:] p_priors):

    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]
    cdef int K = old_p.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] new_p = np.zeros((K, D))

    cdef np.float64_t weight_sum;
    cdef np.float64_t zw;

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
                assert new_p[k, d] - 1.0 < 1e-10
                new_p[k, d] = 1.0
    return new_p


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef log_support_c(np.ndarray[np.uint8_t, ndim=2, cast=True] observations,
                np.ndarray[np.uint8_t, ndim=2, cast=True] not_null_mask,
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
    cdef np.ndarray[np.float64_t, ndim=2] log_pi = np.log(pi)
    cdef np.ndarray[np.float64_t, ndim=2] log_p = np.log(p)

    cdef np.ndarray[np.float64_t, ndim=2] one_minus_p = np.ones((K, D), dtype=np.float64) - p
    cdef np.ndarray[np.float64_t, ndim=2] log_one_minus_p = np.log(one_minus_p)
    #
    # for k in range(K):
    #     for d in range(D):
    #         try:
    #             assert not np.isnan(p[k, d])
    #             assert not np.isnan(log_p[k, d])
    #             assert not np.isnan(log_one_minus_p[k, d])
    #         except AssertionError:
    #             print('p', p[k,d])
    #             print('log_p', log_p[k,d])
    #             print('log_1-p', log_one_minus_p[k,d])
    #             print('1-p', one_minus_p[k,d])
    #             raise

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
