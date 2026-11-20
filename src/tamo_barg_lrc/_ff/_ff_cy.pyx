# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython accelerator wrapping _ff_core.h for GF(2^m) linear algebra and RS decoding."""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t

cnp.import_array()

cdef extern from "_ff_core.h":
    void gf_matvec_u16(
        const uint16_t *H,
        const uint16_t *v,
        uint16_t *out,
        int m,
        int n,
        const uint16_t *exp_tbl,
        const uint16_t *log_tbl,
        uint32_t q,
    ) noexcept nogil

    void gf_matmul_u16(
        const uint16_t *A,
        const uint16_t *B,
        uint16_t *out,
        int m,
        int k,
        int n,
        const uint16_t *exp_tbl,
        const uint16_t *log_tbl,
        uint32_t q,
    ) noexcept nogil

    int gf_rref_u16(
        uint16_t *A,
        int nrows,
        int ncols,
        int32_t *piv_cols,
        const uint16_t *exp_tbl,
        const uint16_t *log_tbl,
        uint32_t q,
    ) noexcept nogil

    int gf_left_kernel_u16(
        const uint16_t *A,
        int nrows,
        int ncols,
        uint16_t *K_out,
        const uint16_t *exp_tbl,
        const uint16_t *log_tbl,
        uint32_t q,
    ) noexcept nogil

    void gf_poly_eval_batch_u16(
        const uint16_t *coeffs,
        int deg_p1,
        const uint16_t *xs,
        int n,
        uint16_t *out,
        const uint16_t *exp_tbl,
        const uint16_t *log_tbl,
        uint32_t q,
    ) noexcept nogil

    int gf_rs_decode_ee_u16(
        const uint16_t *r,
        const uint8_t *era_mask,
        const uint16_t *alphas,
        const uint16_t *v_mult,
        int n,
        int k,
        uint16_t *out,
        const uint16_t *exp_tbl,
        const uint16_t *log_tbl,
        uint32_t q,
    ) noexcept nogil

    void gf_rs_decode_batch_u16(
        const uint16_t *r_batch,
        const uint16_t *alphas,
        const uint16_t *v_mult,
        int B,
        int n,
        int k,
        uint16_t *out_batch,
        uint8_t *ok_mask,
        const uint16_t *exp_tbl,
        const uint16_t *log_tbl,
        uint32_t q,
    ) noexcept nogil


cdef inline cnp.ndarray _as_u16_1d(object obj):
    cdef cnp.ndarray a
    if isinstance(obj, np.ndarray):
        a = <cnp.ndarray>obj
        if (
            cnp.PyArray_NDIM(a) == 1
            and cnp.PyArray_TYPE(a) == cnp.NPY_UINT16
            and cnp.PyArray_IS_C_CONTIGUOUS(a)
        ):
            return a
    a = np.ascontiguousarray(obj, dtype=np.uint16)
    if cnp.PyArray_NDIM(a) != 1:
        raise ValueError(f"Expected 1D array, got {cnp.PyArray_NDIM(a)}D")
    return a


cdef inline cnp.ndarray _as_u16_2d(object obj):
    cdef cnp.ndarray a
    if isinstance(obj, np.ndarray):
        a = <cnp.ndarray>obj
        if (
            cnp.PyArray_NDIM(a) == 2
            and cnp.PyArray_TYPE(a) == cnp.NPY_UINT16
            and cnp.PyArray_IS_C_CONTIGUOUS(a)
        ):
            return a
    a = np.ascontiguousarray(obj, dtype=np.uint16)
    if cnp.PyArray_NDIM(a) != 2:
        raise ValueError(f"Expected 2D array, got {cnp.PyArray_NDIM(a)}D")
    return a


cdef inline cnp.ndarray _as_u8_1d(object obj):
    cdef cnp.ndarray a
    if isinstance(obj, np.ndarray):
        a = <cnp.ndarray>obj
        if (
            cnp.PyArray_NDIM(a) == 1
            and cnp.PyArray_TYPE(a) == cnp.NPY_UINT8
            and cnp.PyArray_IS_C_CONTIGUOUS(a)
        ):
            return a
    a = np.ascontiguousarray(obj, dtype=np.uint8)
    if cnp.PyArray_NDIM(a) != 1:
        raise ValueError(f"Expected 1D array, got {cnp.PyArray_NDIM(a)}D")
    return a


def matvec_cy(H_u16, v_u16, exp_u16, log_u16, uint32_t q):
    """Matrix-vector product H @ v over GF(2^m)."""
    cdef cnp.ndarray H = _as_u16_2d(H_u16)
    cdef cnp.ndarray v = _as_u16_1d(v_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int m = <int>cnp.PyArray_DIM(H, 0)
    cdef int n = <int>cnp.PyArray_DIM(H, 1)
    if cnp.PyArray_DIM(v, 0) != n:
        raise ValueError(f"Shape mismatch: H is ({m}, {n}), v has length {cnp.PyArray_DIM(v, 0)}")
    cdef cnp.ndarray out = np.empty(m, dtype=np.uint16)
    cdef const uint16_t *H_ptr = <const uint16_t *>cnp.PyArray_DATA(H)
    cdef const uint16_t *v_ptr = <const uint16_t *>cnp.PyArray_DATA(v)
    cdef uint16_t *out_ptr = <uint16_t *>cnp.PyArray_DATA(out)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    with nogil:
        gf_matvec_u16(H_ptr, v_ptr, out_ptr, m, n, exp_ptr, log_ptr, q)
    return out


def matmul_cy(A_u16, B_u16, exp_u16, log_u16, uint32_t q):
    """Matrix-matrix product A @ B over GF(2^m)."""
    cdef cnp.ndarray A = _as_u16_2d(A_u16)
    cdef cnp.ndarray B = _as_u16_2d(B_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int m = <int>cnp.PyArray_DIM(A, 0)
    cdef int k = <int>cnp.PyArray_DIM(A, 1)
    cdef int n = <int>cnp.PyArray_DIM(B, 1)
    if cnp.PyArray_DIM(B, 0) != k:
        raise ValueError(
            f"Shape mismatch: A is ({m}, {k}), B is ({cnp.PyArray_DIM(B, 0)}, {n})"
        )
    cdef cnp.ndarray out = np.empty((m, n), dtype=np.uint16)
    cdef const uint16_t *A_ptr = <const uint16_t *>cnp.PyArray_DATA(A)
    cdef const uint16_t *B_ptr = <const uint16_t *>cnp.PyArray_DATA(B)
    cdef uint16_t *out_ptr = <uint16_t *>cnp.PyArray_DATA(out)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    with nogil:
        gf_matmul_u16(A_ptr, B_ptr, out_ptr, m, k, n, exp_ptr, log_ptr, q)
    return out


def rref_cy(A_u16, exp_u16, log_u16, uint32_t q):
    """Reduced row echelon form of A over GF(2^m). Returns (rref_mat, pivot_cols)."""
    cdef cnp.ndarray A_in = _as_u16_2d(A_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int nrows = <int>cnp.PyArray_DIM(A_in, 0)
    cdef int ncols = <int>cnp.PyArray_DIM(A_in, 1)
    cdef cnp.ndarray A = np.array(A_in, dtype=np.uint16, order="C", copy=True)
    cdef int min_d = nrows if nrows < ncols else ncols
    cdef cnp.ndarray pivs = np.empty(min_d, dtype=np.int32)
    cdef uint16_t *A_ptr = <uint16_t *>cnp.PyArray_DATA(A)
    cdef int32_t *piv_ptr = <int32_t *>cnp.PyArray_DATA(pivs)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    cdef int rank
    with nogil:
        rank = gf_rref_u16(A_ptr, nrows, ncols, piv_ptr, exp_ptr, log_ptr, q)
    return A, pivs[:rank].copy()


def left_kernel_cy(A_u16, exp_u16, log_u16, uint32_t q):
    """Basis for the left kernel {x : x @ A == 0} over GF(2^m)."""
    cdef cnp.ndarray A = _as_u16_2d(A_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int nrows = <int>cnp.PyArray_DIM(A, 0)
    cdef int ncols = <int>cnp.PyArray_DIM(A, 1)
    if nrows == 0:
        return np.empty((0, 0), dtype=np.uint16)
    cdef cnp.ndarray K_full = np.empty((nrows, nrows), dtype=np.uint16)
    cdef const uint16_t *A_ptr = <const uint16_t *>cnp.PyArray_DATA(A)
    cdef uint16_t *K_ptr = <uint16_t *>cnp.PyArray_DATA(K_full)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    cdef int nfree
    with nogil:
        nfree = gf_left_kernel_u16(A_ptr, nrows, ncols, K_ptr, exp_ptr, log_ptr, q)
    if nfree < 0:
        raise MemoryError("Failed to allocate scratch buffer in left_kernel_cy")
    if nfree == nrows:
        return K_full
    return np.ascontiguousarray(K_full[:nfree])


def poly_eval_batch_cy(coeffs_u16, xs_u16, exp_u16, log_u16, uint32_t q):
    """Evaluate polynomial with ascending coefficients at xs over GF(2^m)."""
    cdef cnp.ndarray coeffs = _as_u16_1d(coeffs_u16)
    cdef cnp.ndarray xs = _as_u16_1d(xs_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int deg_p1 = <int>cnp.PyArray_DIM(coeffs, 0)
    cdef int n = <int>cnp.PyArray_DIM(xs, 0)
    cdef cnp.ndarray out = np.empty(n, dtype=np.uint16)
    cdef const uint16_t *c_ptr = <const uint16_t *>cnp.PyArray_DATA(coeffs)
    cdef const uint16_t *x_ptr = <const uint16_t *>cnp.PyArray_DATA(xs)
    cdef uint16_t *out_ptr = <uint16_t *>cnp.PyArray_DATA(out)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    with nogil:
        gf_poly_eval_batch_u16(c_ptr, deg_p1, x_ptr, n, out_ptr, exp_ptr, log_ptr, q)
    return out


def rs_decode_syndrome_cy(r_u16, alphas_u16, v_mult_u16, int k, exp_u16, log_u16, uint32_t q):
    """Pure-error Reed-Solomon syndrome decoder. Returns decoded codeword or None."""
    cdef cnp.ndarray r = _as_u16_1d(r_u16)
    cdef cnp.ndarray alphas = _as_u16_1d(alphas_u16)
    cdef cnp.ndarray v_mult = _as_u16_1d(v_mult_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int n = <int>cnp.PyArray_DIM(r, 0)
    if cnp.PyArray_DIM(alphas, 0) != n or cnp.PyArray_DIM(v_mult, 0) != n:
        raise ValueError("r, alphas, and v_mult must have the same length")
    cdef cnp.ndarray out = np.empty(n, dtype=np.uint16)
    cdef const uint16_t *r_ptr = <const uint16_t *>cnp.PyArray_DATA(r)
    cdef const uint16_t *a_ptr = <const uint16_t *>cnp.PyArray_DATA(alphas)
    cdef const uint16_t *v_ptr = <const uint16_t *>cnp.PyArray_DATA(v_mult)
    cdef uint16_t *out_ptr = <uint16_t *>cnp.PyArray_DATA(out)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    cdef int st
    with nogil:
        st = gf_rs_decode_ee_u16(
            r_ptr, NULL, a_ptr, v_ptr, n, k, out_ptr, exp_ptr, log_ptr, q
        )
    if st != 0:
        return None
    return out


def rs_decode_ee_cy(r_u16, era_u8, alphas_u16, v_mult_u16, int k, exp_u16, log_u16, uint32_t q):
    """Error-and-erasure Reed-Solomon decoder. Returns decoded codeword or None."""
    cdef cnp.ndarray r = _as_u16_1d(r_u16)
    cdef cnp.ndarray alphas = _as_u16_1d(alphas_u16)
    cdef cnp.ndarray v_mult = _as_u16_1d(v_mult_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int n = <int>cnp.PyArray_DIM(r, 0)
    if cnp.PyArray_DIM(alphas, 0) != n or cnp.PyArray_DIM(v_mult, 0) != n:
        raise ValueError("r, alphas, and v_mult must have the same length")

    cdef cnp.ndarray era
    cdef const uint8_t *era_ptr = NULL
    if era_u8 is not None:
        era = _as_u8_1d(era_u8)
        if cnp.PyArray_DIM(era, 0) != n:
            raise ValueError("era_u8 must have the same length as r")
        era_ptr = <const uint8_t *>cnp.PyArray_DATA(era)

    cdef cnp.ndarray out = np.empty(n, dtype=np.uint16)
    cdef const uint16_t *r_ptr = <const uint16_t *>cnp.PyArray_DATA(r)
    cdef const uint16_t *a_ptr = <const uint16_t *>cnp.PyArray_DATA(alphas)
    cdef const uint16_t *v_ptr = <const uint16_t *>cnp.PyArray_DATA(v_mult)
    cdef uint16_t *out_ptr = <uint16_t *>cnp.PyArray_DATA(out)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    cdef int st
    with nogil:
        st = gf_rs_decode_ee_u16(
            r_ptr, era_ptr, a_ptr, v_ptr, n, k, out_ptr, exp_ptr, log_ptr, q
        )
    if st != 0:
        return None
    return out


def rs_decode_batch_cy(
    r_batch_u16, alphas_u16, v_mult_u16, int k, exp_u16, log_u16, uint32_t q
):
    """Batch pure-error Reed-Solomon decoder. Returns (decoded_batch, success_mask)."""
    cdef cnp.ndarray r_batch = _as_u16_2d(r_batch_u16)
    cdef cnp.ndarray alphas = _as_u16_1d(alphas_u16)
    cdef cnp.ndarray v_mult = _as_u16_1d(v_mult_u16)
    cdef cnp.ndarray exp_a = _as_u16_1d(exp_u16)
    cdef cnp.ndarray log_a = _as_u16_1d(log_u16)
    cdef int B = <int>cnp.PyArray_DIM(r_batch, 0)
    cdef int n = <int>cnp.PyArray_DIM(r_batch, 1)
    if cnp.PyArray_DIM(alphas, 0) != n or cnp.PyArray_DIM(v_mult, 0) != n:
        raise ValueError("r_batch columns, alphas, and v_mult must have the same length")

    cdef cnp.ndarray out = np.empty((B, n), dtype=np.uint16)
    cdef cnp.ndarray ok_mask = np.empty(B, dtype=np.uint8)
    cdef const uint16_t *rb_ptr = <const uint16_t *>cnp.PyArray_DATA(r_batch)
    cdef const uint16_t *a_ptr = <const uint16_t *>cnp.PyArray_DATA(alphas)
    cdef const uint16_t *v_ptr = <const uint16_t *>cnp.PyArray_DATA(v_mult)
    cdef uint16_t *out_ptr = <uint16_t *>cnp.PyArray_DATA(out)
    cdef uint8_t *ok_ptr = <uint8_t *>cnp.PyArray_DATA(ok_mask)
    cdef const uint16_t *exp_ptr = <const uint16_t *>cnp.PyArray_DATA(exp_a)
    cdef const uint16_t *log_ptr = <const uint16_t *>cnp.PyArray_DATA(log_a)
    with nogil:
        gf_rs_decode_batch_u16(
            rb_ptr, a_ptr, v_ptr, B, n, k, out_ptr, ok_ptr, exp_ptr, log_ptr, q
        )
    return out, ok_mask
