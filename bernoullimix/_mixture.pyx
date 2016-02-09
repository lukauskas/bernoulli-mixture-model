import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef p_update(np.ndarray[np.uint8_t, ndim=2, cast=True] observations,
               np.ndarray[np.uint8_t, ndim=2, cast=True] not_null_mask,
               np.float64_t[::1,:] zstar,
               np.float64_t[:] weight,
               np.float64_t[:,:] old_p):

    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]
    cdef int K = old_p.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] new_p = np.zeros((K, D))

    cdef np.float64_t weight_sum;
    cdef np.float64_t zstar_times_weight;

    for k in range(K):
        weight_sum = 0.0
        for n in range(N):
            zstar_times_weight = zstar[n, k] * weight[n]
            weight_sum += zstar_times_weight

            for d in range(D):
                if not_null_mask[n, d]:
                    new_p[k, d] += observations[n, d] * zstar_times_weight
                else:
                    new_p[k, d] += old_p[k, d] * zstar_times_weight

        for d in range(D):
            new_p[k, d] /= weight_sum

    return new_p


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef support_c(np.ndarray[np.uint8_t, ndim=2, cast=True] observations,
                np.ndarray[np.uint8_t, ndim=2, cast=True] not_null_mask,
                np.int_t[:] dataset_ids_as_ilocs,
                np.float64_t[:,:] pi,
                np.float64_t[:,:] p):


    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]
    cdef int K = pi.shape[1]
    cdef int n_datasets = pi.shape[0]
    cdef np.float64_t row_ans

    cdef np.ndarray[np.float64_t, ndim=2] ans = np.empty((N, K), dtype=float, order='F')

    for k in range(K):
        for n in range(N):

            row_ans = 1
            for d in range(D):
                if not_null_mask[n, d]:
                    if observations[n, d]:
                        row_ans *= p[k, d]
                    else:
                        row_ans *= 1-p[k, d]

            ans[n, k] = row_ans * pi[dataset_ids_as_ilocs[n], k]

    return ans
