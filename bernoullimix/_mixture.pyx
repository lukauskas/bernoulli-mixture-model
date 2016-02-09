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
cpdef partial_support(np.ndarray[np.uint8_t, ndim=2, cast=True] observations,
                      np.ndarray[np.uint8_t, ndim=2, cast=True] not_null_mask,
                      np.float64_t[:] p_k):


    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] ans = np.empty(N, order='F', dtype=np.float64)
    cdef np.float64_t row_ans

    for n in range(N):
        row_ans = 1
        for d in range(D):
            if not_null_mask[n, d]:
                if observations[n, d]:
                    row_ans *= p_k[d]
                else:
                    row_ans *= 1-p_k[d]

        ans[n] = row_ans

    return ans
