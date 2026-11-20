/*
 * _ff_core.h — Portable C99 GF(2^m) finite-field, linear-algebra, polynomial,
 * and Reed-Solomon error-and-erasure decoding kernels for tamo-barg-lrc.
 *
 * All functions operate on uint16_t field elements for q = 2^m <= 65536
 * using caller-provided lookup tables:
 *   - exp_tbl: length >= 2 * (q - 1), exp_tbl[i] = alpha^(i mod (q - 1))
 *   - log_tbl: length == q, log_tbl[x] in [0, q - 2] for x != 0
 *
 * Zero external C dependencies; compiles cleanly with -std=c99 -Wall -Wextra.
 */

#ifndef TAMO_BARG_LRC_FF_CORE_H
#define TAMO_BARG_LRC_FF_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FF_CORE_STACK_N 1024
#define FF_CORE_STACK_M 512

/* --------------------------------------------------------------------------
 * Branchless LUT GF(2^m) scalar arithmetic (q = 2^m <= 65536)
 * -------------------------------------------------------------------------- */

static inline uint16_t gf_mul(
    uint16_t a,
    uint16_t b,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl
) {
    uint32_t mask = (uint32_t)(-(int32_t)((a != 0u) & (b != 0u)));
    uint32_t idx = ((uint32_t)log_tbl[a] + (uint32_t)log_tbl[b]) & mask;
    return (uint16_t)(exp_tbl[idx] & mask);
}

static inline uint16_t gf_inv(
    uint16_t a,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    uint32_t mask = (uint32_t)(-(int32_t)(a != 0u));
    uint32_t idx = ((q - 1u) - (uint32_t)log_tbl[a]) & mask;
    return (uint16_t)(exp_tbl[idx] & mask);
}

static inline uint16_t gf_div(
    uint16_t a,
    uint16_t b,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    uint32_t mask = (uint32_t)(-(int32_t)((a != 0u) & (b != 0u)));
    uint32_t idx = ((q - 1u) + (uint32_t)log_tbl[a] - (uint32_t)log_tbl[b]) & mask;
    return (uint16_t)(exp_tbl[idx] & mask);
}

static inline uint16_t gf_pow(
    uint16_t a,
    int64_t e,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    if (e == 0) {
        return 1u;
    }
    if (a == 0u) {
        return 0u;
    }
    int64_t qm1 = (int64_t)(q - 1u);
    int64_t em = e % qm1;
    if (em < 0) {
        em += qm1;
    }
    uint32_t idx = (uint32_t)(((uint64_t)log_tbl[a] * (uint64_t)em) % (uint64_t)(q - 1u));
    return exp_tbl[idx];
}

/* Return canonical discrete log of a^{-1} in [0, q - 2] for nonzero a. */
static inline uint32_t gf_inv_log(uint16_t a, const uint16_t *log_tbl, uint32_t q) {
    uint32_t la = (uint32_t)log_tbl[a];
    return (la == 0u) ? 0u : ((q - 1u) - la);
}

/* --------------------------------------------------------------------------
 * Matrix-vector and matrix-matrix multiplication over GF(2^m)
 * -------------------------------------------------------------------------- */

static inline void gf_matvec_u16(
    const uint16_t *H,
    const uint16_t *v,
    uint16_t *out,
    int m,
    int n,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    (void)q;
    if (m <= 0) {
        return;
    }
    if (n <= 0) {
        memset(out, 0, (size_t)m * sizeof(uint16_t));
        return;
    }

    int nz_idx_stack[FF_CORE_STACK_N];
    uint16_t nz_log_stack[FF_CORE_STACK_N];
    int *nz_idx = nz_idx_stack;
    uint16_t *nz_log = nz_log_stack;
    int heap_alloc = 0;

    if (n > FF_CORE_STACK_N) {
        nz_idx = (int *)malloc((size_t)n * sizeof(int));
        nz_log = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
        if (!nz_idx || !nz_log) {
            free(nz_idx);
            free(nz_log);
            for (int i = 0; i < m; i++) {
                const uint16_t *hi = H + (size_t)i * (size_t)n;
                uint16_t acc = 0u;
                for (int j = 0; j < n; j++) {
                    uint16_t hij = hi[j];
                    uint16_t vj = v[j];
                    if (hij != 0u && vj != 0u) {
                        acc ^= exp_tbl[(uint32_t)log_tbl[hij] + (uint32_t)log_tbl[vj]];
                    }
                }
                out[i] = acc;
            }
            return;
        }
        heap_alloc = 1;
    }

    int nnz = 0;
    for (int j = 0; j < n; j++) {
        uint16_t vj = v[j];
        if (vj != 0u) {
            nz_idx[nnz] = j;
            nz_log[nnz] = log_tbl[vj];
            nnz++;
        }
    }

    if (nnz == 0) {
        memset(out, 0, (size_t)m * sizeof(uint16_t));
    } else {
        for (int i = 0; i < m; i++) {
            const uint16_t *hi = H + (size_t)i * (size_t)n;
            uint16_t acc = 0u;
            for (int t = 0; t < nnz; t++) {
                uint16_t hij = hi[nz_idx[t]];
                if (hij != 0u) {
                    acc ^= exp_tbl[(uint32_t)log_tbl[hij] + (uint32_t)nz_log[t]];
                }
            }
            out[i] = acc;
        }
    }

    if (heap_alloc) {
        free(nz_idx);
        free(nz_log);
    }
}

static inline void gf_matmul_u16(
    const uint16_t *A,
    const uint16_t *B,
    uint16_t *out,
    int m,
    int k,
    int n,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    (void)q;
    if (m <= 0 || n <= 0) {
        return;
    }
    memset(out, 0, (size_t)m * (size_t)n * sizeof(uint16_t));
    if (k <= 0) {
        return;
    }

    for (int i = 0; i < m; i++) {
        uint16_t *oi = out + (size_t)i * (size_t)n;
        const uint16_t *ai = A + (size_t)i * (size_t)k;
        for (int t = 0; t < k; t++) {
            uint16_t ait = ai[t];
            if (ait == 0u) {
                continue;
            }
            const uint16_t *exp_la = exp_tbl + (uint32_t)log_tbl[ait];
            const uint16_t *bt = B + (size_t)t * (size_t)n;
            for (int j = 0; j < n; j++) {
                uint16_t btj = bt[j];
                if (btj != 0u) {
                    oi[j] ^= exp_la[log_tbl[btj]];
                }
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * In-place RREF and Left Kernel over GF(2^m)
 * -------------------------------------------------------------------------- */

static inline int gf_rref_u16(
    uint16_t *A,
    int nrows,
    int ncols,
    int32_t *piv_cols,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    if (nrows <= 0 || ncols <= 0) {
        return 0;
    }

    int nz_col_stack[FF_CORE_STACK_N];
    uint16_t nz_log_stack[FF_CORE_STACK_N];
    int *nz_col = nz_col_stack;
    uint16_t *nz_log = nz_log_stack;
    int heap_alloc = 0;

    if (ncols > FF_CORE_STACK_N) {
        nz_col = (int *)malloc((size_t)ncols * sizeof(int));
        nz_log = (uint16_t *)malloc((size_t)ncols * sizeof(uint16_t));
        if (!nz_col || !nz_log) {
            free(nz_col);
            free(nz_log);
            nz_col = NULL;
            nz_log = NULL;
        } else {
            heap_alloc = 1;
        }
    }

    int npiv = 0;
    int r = 0;
    for (int c = 0; c < ncols && r < nrows; c++) {
        int piv_row = -1;
        for (int i = r; i < nrows; i++) {
            if (A[(size_t)i * (size_t)ncols + (size_t)c] != 0u) {
                piv_row = i;
                break;
            }
        }
        if (piv_row == -1) {
            continue;
        }

        uint16_t *rr = A + (size_t)r * (size_t)ncols;
        if (piv_row != r) {
            uint16_t *rb = A + (size_t)piv_row * (size_t)ncols;
            for (int j = c; j < ncols; j++) {
                uint16_t tmp = rr[j];
                rr[j] = rb[j];
                rb[j] = tmp;
            }
        }

        uint16_t pv = rr[c];
        if (pv != 1u) {
            const uint16_t *exp_inv = exp_tbl + gf_inv_log(pv, log_tbl, q);
            rr[c] = 1u;
            for (int j = c + 1; j < ncols; j++) {
                uint16_t v = rr[j];
                if (v != 0u) {
                    rr[j] = exp_inv[log_tbl[v]];
                }
            }
        }

        if (nz_col != NULL) {
            int nnz = 0;
            for (int j = c + 1; j < ncols; j++) {
                uint16_t v = rr[j];
                if (v != 0u) {
                    nz_col[nnz] = j;
                    nz_log[nnz] = log_tbl[v];
                    nnz++;
                }
            }
            for (int i = 0; i < nrows; i++) {
                if (i == r) {
                    continue;
                }
                uint16_t *ri = A + (size_t)i * (size_t)ncols;
                uint16_t fac = ri[c];
                if (fac != 0u) {
                    ri[c] = 0u;
                    const uint16_t *exp_fac = exp_tbl + (uint32_t)log_tbl[fac];
                    for (int t = 0; t < nnz; t++) {
                        ri[nz_col[t]] ^= exp_fac[nz_log[t]];
                    }
                }
            }
        } else {
            for (int i = 0; i < nrows; i++) {
                if (i == r) {
                    continue;
                }
                uint16_t *ri = A + (size_t)i * (size_t)ncols;
                uint16_t fac = ri[c];
                if (fac != 0u) {
                    ri[c] = 0u;
                    const uint16_t *exp_fac = exp_tbl + (uint32_t)log_tbl[fac];
                    for (int j = c + 1; j < ncols; j++) {
                        uint16_t rrj = rr[j];
                        if (rrj != 0u) {
                            ri[j] ^= exp_fac[log_tbl[rrj]];
                        }
                    }
                }
            }
        }

        piv_cols[npiv++] = (int32_t)c;
        r++;
    }

    if (heap_alloc) {
        free(nz_col);
        free(nz_log);
    }
    return npiv;
}

/* Compute a basis for the left kernel {x : x A = 0} into K_out (nrows x nrows max).
 * Returns nullity = nrows - rank, or -1 on allocation failure. */
static inline int gf_left_kernel_u16(
    const uint16_t *A,
    int nrows,
    int ncols,
    uint16_t *K_out,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    if (nrows <= 0) {
        return 0;
    }
    if (ncols <= 0) {
        memset(K_out, 0, (size_t)nrows * (size_t)nrows * sizeof(uint16_t));
        for (int i = 0; i < nrows; i++) {
            K_out[(size_t)i * (size_t)nrows + (size_t)i] = 1u;
        }
        return nrows;
    }

    uint16_t *MT = (uint16_t *)malloc((size_t)ncols * (size_t)nrows * sizeof(uint16_t));
    int min_d = nrows < ncols ? nrows : ncols;
    int32_t *pivs = (int32_t *)malloc((size_t)(min_d + 1) * sizeof(int32_t));
    uint8_t *is_piv = (uint8_t *)calloc((size_t)nrows, 1);
    if (!MT || !pivs || !is_piv) {
        free(MT);
        free(pivs);
        free(is_piv);
        return -1;
    }

    for (int i = 0; i < nrows; i++) {
        const uint16_t *ai = A + (size_t)i * (size_t)ncols;
        for (int j = 0; j < ncols; j++) {
            MT[(size_t)j * (size_t)nrows + (size_t)i] = ai[j];
        }
    }

    int npiv = gf_rref_u16(MT, ncols, nrows, pivs, exp_tbl, log_tbl, q);
    for (int i = 0; i < npiv; i++) {
        is_piv[pivs[i]] = 1u;
    }

    int nfree = nrows - npiv;
    if (nfree > 0) {
        memset(K_out, 0, (size_t)nfree * (size_t)nrows * sizeof(uint16_t));
        int fi = 0;
        for (int j = 0; j < nrows; j++) {
            if (!is_piv[j]) {
                uint16_t *krow = K_out + (size_t)fi * (size_t)nrows;
                krow[j] = 1u;
                for (int pi = 0; pi < npiv; pi++) {
                    uint16_t val = MT[(size_t)pi * (size_t)nrows + (size_t)j];
                    if (val != 0u) {
                        krow[pivs[pi]] = val;
                    }
                }
                fi++;
            }
        }
    }

    free(is_piv);
    free(pivs);
    free(MT);
    return nfree;
}

/* --------------------------------------------------------------------------
 * Batch Horner polynomial evaluation over GF(2^m)
 * -------------------------------------------------------------------------- */

static inline void gf_poly_eval_batch_u16(
    const uint16_t *coeffs,
    int deg_p1,
    const uint16_t *xs,
    int n,
    uint16_t *out,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    (void)q;
    if (n <= 0) {
        return;
    }
    int dp = deg_p1 - 1;
    while (dp >= 0 && coeffs[dp] == 0u) {
        dp--;
    }
    if (dp < 0) {
        memset(out, 0, (size_t)n * sizeof(uint16_t));
        return;
    }
    for (int j = 0; j < n; j++) {
        uint16_t xj = xs[j];
        if (xj == 0u) {
            out[j] = coeffs[0];
            continue;
        }
        const uint16_t *exp_lx = exp_tbl + (uint32_t)log_tbl[xj];
        uint16_t acc = coeffs[dp];
        for (int i = dp - 1; i >= 0; i--) {
            acc = (acc != 0u ? exp_lx[log_tbl[acc]] : 0u) ^ coeffs[i];
        }
        out[j] = acc;
    }
}

/* --------------------------------------------------------------------------
 * Unified Sugiyama + Chien + Forney RS error-and-erasure decoder over GF(2^m)
 *
 * Parity-check matrix is implicitly H[i, j] = v_mult[j] * alphas[j]^i
 * for 0 <= i < n - k, 0 <= j < n, where v_mult[j] = 1 / prod_{l != j}(a_j - a_l).
 * Handles arbitrary distinct evaluation points alphas (including 0), pure errors
 * (era_mask == NULL or all zero), combined errors+erasures, pure erasures up to
 * d - 1, d = 2, and n = k.
 *
 * Performs mandatory full syndrome self-verification before returning 0.
 * Returns 0 on success, -1 on DecodingError.
 * -------------------------------------------------------------------------- */

static inline int gf_rs_decode_ee_u16(
    const uint16_t *r,
    const uint8_t *era_mask,
    const uint16_t *alphas,
    const uint16_t *v_mult,
    int n,
    int k,
    uint16_t *out,
    const uint16_t *exp_tbl,
    const uint16_t *log_tbl,
    uint32_t q
) {
    if (n < 0 || k < 0 || k > n) {
        return -1;
    }
    if (n == 0) {
        return 0;
    }

    int m = n - k;
    int d = m + 1;
    memcpy(out, r, (size_t)n * sizeof(uint16_t));

    int e_cnt = 0;
    if (era_mask != NULL) {
        for (int j = 0; j < n; j++) {
            if (era_mask[j] != 0u) {
                out[j] = 0u;
                e_cnt++;
            }
        }
        if (e_cnt >= d) {
            return -1;
        }
    }

    if (m == 0) {
        return 0;
    }

    /* Locate evaluation point alpha == 0, if present. */
    int zero_idx = -1;
    for (int j = 0; j < n; j++) {
        if (alphas[j] == 0u) {
            zero_idx = j;
            break;
        }
    }

    /* Scratch memory: stack for m <= FF_CORE_STACK_M, single heap block otherwise. */
    uint16_t u16_stack[12 * (FF_CORE_STACK_M + 2)];
    int int_stack[3 * (FF_CORE_STACK_M + 2)];
    uint16_t *u16_buf = u16_stack;
    int *int_buf = int_stack;
    int heap_alloc = 0;
    size_t stride = (size_t)(m + 2);

    if (m > FF_CORE_STACK_M) {
        u16_buf = (uint16_t *)malloc(12u * stride * sizeof(uint16_t));
        int_buf = (int *)malloc(3u * stride * sizeof(int));
        if (!u16_buf || !int_buf) {
            free(u16_buf);
            free(int_buf);
            return -1;
        }
        heap_alloc = 1;
    }

    uint16_t *S       = u16_buf + 0u * stride;
    uint16_t *S_chk   = u16_buf + 1u * stride;
    uint16_t *gamma   = u16_buf + 2u * stride;
    uint16_t *xi      = u16_buf + 3u * stride;
    uint16_t *r0      = u16_buf + 4u * stride;
    uint16_t *r1      = u16_buf + 5u * stride;
    uint16_t *t0      = u16_buf + 6u * stride;
    uint16_t *t1      = u16_buf + 7u * stride;
    uint16_t *qb      = u16_buf + 8u * stride;
    uint16_t *t2      = u16_buf + 9u * stride;
    uint16_t *psi     = u16_buf + 10u * stride;
    uint16_t *act_val = u16_buf + 11u * stride;

    int *u_star  = int_buf + 0u * stride;
    int *e_star  = int_buf + 1u * stride;
    int *act_idx = int_buf + 2u * stride;

    int status = -1;

    /* 1. Compute syndromes S[i] = sum_j v_mult[j] * alphas[j]^i * out[j]. */
    memset(S, 0, (size_t)m * sizeof(uint16_t));
    for (int j = 0; j < n; j++) {
        uint16_t rj = out[j];
        if (rj == 0u) {
            continue;
        }
        uint16_t vj = v_mult[j];
        if (vj == 0u) {
            continue;
        }
        uint16_t p = exp_tbl[(uint32_t)log_tbl[vj] + (uint32_t)log_tbl[rj]];
        uint16_t aj = alphas[j];
        if (aj == 0u) {
            S[0] ^= p;
        } else {
            const uint16_t *exp_a = exp_tbl + (uint32_t)log_tbl[aj];
            S[0] ^= p;
            for (int i = 1; i < m; i++) {
                p = exp_a[log_tbl[p]];
                S[i] ^= p;
            }
        }
    }

    int all_zero = 1;
    for (int i = 0; i < m; i++) {
        if (S[i] != 0u) {
            all_zero = 0;
            break;
        }
    }
    if (all_zero) {
        status = 0;
        goto cleanup;
    }

    int T = (m - e_cnt) / 2;
    if (T == 0 && e_cnt == 0) {
        goto cleanup;
    }

    /* 2. Build erasure locator gamma(x) = prod_{j in U*} (1 + alpha_j x). */
    memset(gamma, 0, (size_t)(e_cnt + 2) * sizeof(uint16_t));
    gamma[0] = 1u;
    int gdeg = 0;
    int nu_star = 0;
    if (era_mask != NULL && e_cnt > 0) {
        for (int j = 0; j < n; j++) {
            if (era_mask[j] != 0u && j != zero_idx) {
                u_star[nu_star++] = j;
                const uint16_t *exp_la = exp_tbl + (uint32_t)log_tbl[alphas[j]];
                for (int di = gdeg; di >= 0; di--) {
                    uint16_t gdi = gamma[di];
                    if (gdi != 0u) {
                        gamma[di + 1] ^= exp_la[log_tbl[gdi]];
                    }
                }
                gdeg++;
            }
        }
    }

    /* 3. Forney modified syndrome xi(x) = gamma(x) * S(x) mod x^m. */
    memset(xi, 0, (size_t)m * sizeof(uint16_t));
    for (int i = 0; i <= gdeg; i++) {
        uint16_t gi = gamma[i];
        if (gi == 0u) {
            continue;
        }
        const uint16_t *exp_lg = exp_tbl + (uint32_t)log_tbl[gi];
        int limit = m - i;
        for (int j = 0; j < limit; j++) {
            uint16_t sj = S[j];
            if (sj != 0u) {
                xi[i + j] ^= exp_lg[log_tbl[sj]];
            }
        }
    }

    /* 4. Sugiyama extended Euclidean algorithm on (x^m, xi(x)). */
    memset(r0, 0, stride * sizeof(uint16_t));
    memset(r1, 0, stride * sizeof(uint16_t));
    memset(t0, 0, stride * sizeof(uint16_t));
    memset(t1, 0, stride * sizeof(uint16_t));

    r0[m] = 1u;
    int dr0 = m;
    int dr1 = -1;
    for (int i = 0; i < m; i++) {
        r1[i] = xi[i];
        if (xi[i] != 0u) {
            dr1 = i;
        }
    }
    int dt0 = -1;
    t1[0] = 1u;
    int dt1 = 0;
    int stop_deg = T + e_cnt;

    while (dr1 >= stop_deg) {
        int dq = dr0 - dr1;
        memset(qb, 0, (size_t)(dq + 1) * sizeof(uint16_t));
        const uint16_t *exp_linv = exp_tbl + gf_inv_log(r1[dr1], log_tbl, q);
        for (int shift = dq; shift >= 0; shift--) {
            uint16_t c = r0[shift + dr1];
            if (c != 0u) {
                uint16_t fac = exp_linv[log_tbl[c]];
                qb[shift] = fac;
                const uint16_t *exp_lfac = exp_tbl + (uint32_t)log_tbl[fac];
                for (int j = 0; j <= dr1; j++) {
                    uint16_t r1j = r1[j];
                    if (r1j != 0u) {
                        r0[shift + j] ^= exp_lfac[log_tbl[r1j]];
                    }
                }
            }
        }
        int dr2 = dr1 - 1;
        while (dr2 >= 0 && r0[dr2] == 0u) {
            dr2--;
        }
        int dt2 = dt0 > (dq + dt1) ? dt0 : (dq + dt1);
        for (int i = 0; i <= dt2; i++) {
            t2[i] = (i <= dt0) ? t0[i] : 0u;
        }
        for (int i = 0; i <= dq; i++) {
            uint16_t qbi = qb[i];
            if (qbi == 0u) {
                continue;
            }
            const uint16_t *exp_lq = exp_tbl + (uint32_t)log_tbl[qbi];
            for (int j = 0; j <= dt1; j++) {
                uint16_t t1j = t1[j];
                if (t1j != 0u) {
                    t2[i + j] ^= exp_lq[log_tbl[t1j]];
                }
            }
        }
        while (dt2 >= 0 && t2[dt2] == 0u) {
            dt2--;
        }

        uint16_t *tmp = r0;
        r0 = r1;
        r1 = tmp;
        dr0 = dr1;
        dr1 = dr2;

        tmp = t0;
        t0 = t1;
        t1 = t2;
        t2 = tmp;
        dt0 = dt1;
        dt1 = dt2;
    }

    if (dt1 < 0 || t1[0] == 0u || dt1 > T) {
        goto cleanup;
    }

    /* Normalize so Lambda(0) == 1. */
    if (t1[0] != 1u) {
        const uint16_t *exp_linv0 = exp_tbl + gf_inv_log(t1[0], log_tbl, q);
        for (int i = 0; i <= dt1; i++) {
            if (t1[i] != 0u) {
                t1[i] = exp_linv0[log_tbl[t1[i]]];
            }
        }
        for (int i = 0; i <= dr1; i++) {
            if (r1[i] != 0u) {
                r1[i] = exp_linv0[log_tbl[r1[i]]];
            }
        }
    }

    /* 5. Chien search for nonzero error locations (roots of Lambda at alpha_j^{-1}). */
    int ne_star = 0;
    if (dt1 > 0) {
        for (int j = 0; j < n; j++) {
            if (j == zero_idx) {
                continue;
            }
            if (era_mask != NULL && era_mask[j] != 0u) {
                continue;
            }
            const uint16_t *exp_lx = exp_tbl + gf_inv_log(alphas[j], log_tbl, q);
            uint16_t acc = t1[dt1];
            for (int di = dt1 - 1; di >= 0; di--) {
                acc = (acc != 0u ? exp_lx[log_tbl[acc]] : 0u) ^ t1[di];
            }
            if (acc == 0u) {
                if (ne_star >= T) {
                    goto cleanup;
                }
                e_star[ne_star++] = j;
            }
        }
        if (ne_star != dt1) {
            goto cleanup;
        }
    }

    /* 6. Combined errata locator psi(x) = gamma(x) * Lambda(x). */
    int dpsi = gdeg + dt1;
    memset(psi, 0, (size_t)(dpsi + 1) * sizeof(uint16_t));
    for (int i = 0; i <= gdeg; i++) {
        uint16_t gi = gamma[i];
        if (gi == 0u) {
            continue;
        }
        const uint16_t *exp_lg = exp_tbl + (uint32_t)log_tbl[gi];
        for (int j = 0; j <= dt1; j++) {
            uint16_t t1j = t1[j];
            if (t1j != 0u) {
                psi[i + j] ^= exp_lg[log_tbl[t1j]];
            }
        }
    }

    /* 7. Forney error/erasure magnitudes for all nonzero evaluation points. */
    int nact = nu_star + ne_star;
    int last_odd = (dpsi & 1) ? dpsi : (dpsi - 1);
    for (int idx_i = 0; idx_i < nact; idx_i++) {
        int j = (idx_i < nu_star) ? u_star[idx_i] : e_star[idx_i - nu_star];
        act_idx[idx_i] = j;
        uint32_t lx = gf_inv_log(alphas[j], log_tbl, q);
        const uint16_t *exp_lx = exp_tbl + lx;

        uint16_t om_val = 0u;
        if (dr1 >= 0) {
            om_val = r1[dr1];
            for (int di = dr1 - 1; di >= 0; di--) {
                om_val = (om_val != 0u ? exp_lx[log_tbl[om_val]] : 0u) ^ r1[di];
            }
        }

        uint16_t pd_val = 0u;
        if (last_odd >= 1) {
            uint32_t lx2 = lx + lx;
            if (lx2 >= q - 1u) {
                lx2 -= (q - 1u);
            }
            const uint16_t *exp_lx2 = exp_tbl + lx2;
            pd_val = psi[last_odd];
            for (int di = last_odd - 2; di >= 1; di -= 2) {
                pd_val = (pd_val != 0u ? exp_lx2[log_tbl[pd_val]] : 0u) ^ psi[di];
            }
        }
        if (pd_val == 0u || v_mult[j] == 0u) {
            goto cleanup;
        }
        uint16_t den = exp_tbl[(uint32_t)log_tbl[v_mult[j]] + (uint32_t)log_tbl[pd_val]];
        uint16_t num = gf_mul(alphas[j], om_val, exp_tbl, log_tbl);
        act_val[idx_i] = gf_div(num, den, exp_tbl, log_tbl, q);
    }

    /* 8. Recover error/erasure at alpha == 0 (if present) from S[0]. */
    int total_act = nact;
    int unmarked_nz = 0;
    for (int idx_i = nu_star; idx_i < nact; idx_i++) {
        if (act_val[idx_i] != 0u) {
            unmarked_nz++;
        }
    }
    if (zero_idx >= 0) {
        if (v_mult[zero_idx] == 0u) {
            goto cleanup;
        }
        uint16_t acc0 = 0u;
        for (int idx_i = 0; idx_i < nact; idx_i++) {
            uint16_t ev = act_val[idx_i];
            if (ev != 0u) {
                acc0 ^= gf_mul(ev, v_mult[act_idx[idx_i]], exp_tbl, log_tbl);
            }
        }
        uint16_t rem0 = S[0] ^ acc0;
        uint16_t ej0 = gf_div(rem0, v_mult[zero_idx], exp_tbl, log_tbl, q);
        if (ej0 != 0u) {
            int zero_erased = (era_mask != NULL && era_mask[zero_idx] != 0u);
            if (!zero_erased) {
                unmarked_nz++;
            }
            act_idx[total_act] = zero_idx;
            act_val[total_act] = ej0;
            total_act++;
        }
    }

    if (unmarked_nz > T) {
        goto cleanup;
    }

    /* 9. Mandatory full syndrome self-verification. */
    memset(S_chk, 0, (size_t)m * sizeof(uint16_t));
    for (int idx_i = 0; idx_i < total_act; idx_i++) {
        uint16_t ev = act_val[idx_i];
        if (ev == 0u) {
            continue;
        }
        int j = act_idx[idx_i];
        uint16_t vj = v_mult[j];
        if (vj == 0u) {
            continue;
        }
        uint16_t p = exp_tbl[(uint32_t)log_tbl[vj] + (uint32_t)log_tbl[ev]];
        uint16_t aj = alphas[j];
        if (aj == 0u) {
            S_chk[0] ^= p;
        } else {
            const uint16_t *exp_a = exp_tbl + (uint32_t)log_tbl[aj];
            S_chk[0] ^= p;
            for (int i = 1; i < m; i++) {
                p = exp_a[log_tbl[p]];
                S_chk[i] ^= p;
            }
        }
    }
    if (memcmp(S_chk, S, (size_t)m * sizeof(uint16_t)) != 0) {
        goto cleanup;
    }

    /* 10. Apply verified corrections. */
    for (int idx_i = 0; idx_i < total_act; idx_i++) {
        out[act_idx[idx_i]] ^= act_val[idx_i];
    }
    status = 0;

cleanup:
    if (heap_alloc) {
        free(u16_buf);
        free(int_buf);
    }
    return status;
}

/* --------------------------------------------------------------------------
 * Batch multi-word RS decoder (pure errors, B rows of length n)
 * -------------------------------------------------------------------------- */

static inline void gf_rs_decode_batch_u16(
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
    uint32_t q
) {
    for (int b = 0; b < B; b++) {
        const uint16_t *rb = r_batch + (size_t)b * (size_t)n;
        uint16_t *ob = out_batch + (size_t)b * (size_t)n;
        int st = gf_rs_decode_ee_u16(
            rb, NULL, alphas, v_mult, n, k, ob, exp_tbl, log_tbl, q
        );
        ok_mask[b] = (st == 0) ? 1u : 0u;
    }
}

#endif /* TAMO_BARG_LRC_FF_CORE_H */
