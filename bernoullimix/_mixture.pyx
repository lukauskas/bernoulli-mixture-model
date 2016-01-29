import numpy as np
cimport numpy as np

cpdef zstar_dot_xstar(np.ndarray[np.float64_t, ndim=1] zstar_times_weight_k,
                      np.ndarray[np.uint8_t, ndim=2, cast=True] observations,
                      np.ndarray[np.uint8_t, ndim=2, cast=True] not_null_mask,
                      np.ndarray[np.float64_t, ndim=1] old_p_k):

    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]

    cdef int n;
    cdef int d;

    cdef np.ndarray[np.float64_t, ndim=1] new_p = np.zeros(D)

    for d in range(D):
        for n in range(N):
            if not_null_mask[n, d]:
                new_p[d] += observations[n,d] * zstar_times_weight_k[n]
            else:
                new_p[d] += old_p_k[d] * zstar_times_weight_k[n]

    return new_p

cpdef partial_support(np.ndarray[np.uint8_t, ndim=2, cast=True] observations,
                      np.ndarray[np.uint8_t, ndim=2, cast=True] not_null_mask,
                      np.ndarray[np.float64_t, ndim=1] p_k):


    cdef int N = observations.shape[0]
    cdef int D = observations.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] ans = np.empty(N)
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
