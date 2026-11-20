"""Conway polynomials and finite-field number-theoretic helpers.

All Conway polynomials match Frank Luebeck's canonical database and SageMath's
``conway_polynomials`` table. For any unlisted ``(p, m)``, a deterministic
fallback search finds a monic primitive irreducible polynomial of degree ``m``
over ``GF(p)``.
"""

from __future__ import annotations

import itertools

# Canonical Conway polynomials stored as ascending coefficient tuples
# (c_0, c_1, ..., c_m) representing sum_{i=0}^{m} c_i * x^i over GF(p).
CONWAY_POLYNOMIALS: dict[tuple[int, int], tuple[int, ...]] = {
    # Binary extension fields GF(2^m), m = 1..16
    (2, 1): (1, 1),
    (2, 2): (1, 1, 1),
    (2, 3): (1, 1, 0, 1),
    (2, 4): (1, 1, 0, 0, 1),
    (2, 5): (1, 0, 1, 0, 0, 1),
    (2, 6): (1, 1, 0, 1, 1, 0, 1),
    (2, 7): (1, 1, 0, 0, 0, 0, 0, 1),
    (2, 8): (1, 0, 1, 1, 1, 0, 0, 0, 1),
    (2, 9): (1, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    (2, 10): (1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1),
    (2, 11): (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (2, 12): (1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1),
    (2, 13): (1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (2, 14): (1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1),
    (2, 15): (1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (2, 16): (1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    # Odd prime powers GF(3^2..3^8)
    (3, 2): (2, 2, 1),
    (3, 3): (1, 2, 0, 1),
    (3, 4): (2, 0, 0, 2, 1),
    (3, 5): (1, 2, 0, 0, 0, 1),
    (3, 6): (2, 2, 1, 0, 2, 0, 1),
    (3, 7): (1, 0, 2, 0, 0, 0, 0, 1),
    (3, 8): (2, 2, 2, 0, 1, 2, 0, 0, 1),
    # GF(5^2..5^6)
    (5, 2): (2, 4, 1),
    (5, 3): (3, 3, 0, 1),
    (5, 4): (2, 4, 4, 0, 1),
    (5, 5): (3, 4, 0, 0, 0, 1),
    (5, 6): (2, 0, 1, 4, 1, 0, 1),
    # GF(7^2..7^5)
    (7, 2): (3, 6, 1),
    (7, 3): (4, 0, 6, 1),
    (7, 4): (3, 4, 5, 0, 1),
    (7, 5): (4, 1, 0, 0, 0, 1),
    # GF(11^2..11^4)
    (11, 2): (2, 7, 1),
    (11, 3): (9, 2, 0, 1),
    (11, 4): (2, 10, 8, 0, 1),
    # GF(13^2..13^3)
    (13, 2): (2, 12, 1),
    (13, 3): (11, 2, 0, 1),
    # Quadratic extensions for primes 17..37
    (17, 2): (3, 16, 1),
    (19, 2): (2, 18, 1),
    (23, 2): (5, 21, 1),
    (29, 2): (2, 24, 1),
    (31, 2): (3, 29, 1),
    (37, 2): (2, 33, 1),
}

_FALLBACK_CACHE: dict[tuple[int, int], tuple[int, ...]] = {}


def factor_prime_power(q: int) -> tuple[int, int]:
    """Return ``(p, m)`` such that ``q == p**m`` with ``p`` prime, else raise ``ValueError``."""
    if isinstance(q, bool):
        raise ValueError(f"Field order must be a prime power >= 2, got {q!r}")
    try:
        q_int = int(q)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field order must be a prime power >= 2, got {q!r}") from exc
    if q_int != q or q_int < 2:
        raise ValueError(f"Field order must be a prime power >= 2, got {q!r}")

    p = 2
    while p * p <= q_int:
        if q_int % p == 0:
            break
        p += 1 if p == 2 else 2
    else:
        return q_int, 1

    m = 0
    rem = q_int
    while rem % p == 0:
        rem //= p
        m += 1
    if rem != 1:
        raise ValueError(f"{q_int} is not a prime power")
    return p, m


def divisors(n: int) -> list[int]:
    """Return the sorted list of all positive divisors of integer ``n >= 1``."""
    if isinstance(n, bool):
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    n_int = int(n)
    if n_int < 1:
        raise ValueError(f"n must be >= 1, got {n_int}")
    small: list[int] = []
    large: list[int] = []
    d = 1
    while d * d <= n_int:
        if n_int % d == 0:
            small.append(d)
            if d * d != n_int:
                large.append(n_int // d)
        d += 1
    large.reverse()
    return small + large


def prime_factors(n: int) -> list[int]:
    """Return sorted distinct prime factors of ``n >= 1``."""
    factors: list[int] = []
    d = 2
    rem = int(n)
    while d * d <= rem:
        if rem % d == 0:
            factors.append(d)
            while rem % d == 0:
                rem //= d
        d += 1 if d == 2 else 2
    if rem > 1:
        factors.append(rem)
    return factors


def find_primitive_root_mod_p(p: int) -> int:
    """Return the smallest primitive root modulo prime ``p`` (returns ``1`` for ``p == 2``)."""
    if p == 2:
        return 1
    phi = p - 1
    pf = prime_factors(phi)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in pf):
            return g
    raise RuntimeError(f"No primitive root found modulo {p}")


def _poly_mul_mod_fp(a: list[int], b: list[int], mod: tuple[int, ...], p: int) -> list[int]:
    """Multiply two polynomials over GF(p) modulo monic ``mod`` of degree ``m``."""
    m = len(mod) - 1
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    res[i + j] = (res[i + j] + ai * bj) % p
    for deg in range(len(res) - 1, m - 1, -1):
        c = res[deg]
        if c:
            for j in range(m):
                res[deg - m + j] = (res[deg - m + j] - c * mod[j]) % p
            res[deg] = 0
    return res[:m]


def _x_pow_mod_fp(exp: int, mod: tuple[int, ...], p: int) -> list[int]:
    """Compute ``x**exp mod mod`` in ``GF(p)[x]``."""
    m = len(mod) - 1
    base = [0] * m
    if m == 1:
        base[0] = (-mod[0]) % p
    else:
        base[1] = 1
    acc = [0] * m
    acc[0] = 1
    e = exp
    while e > 0:
        if e & 1:
            acc = _poly_mul_mod_fp(acc, base, mod, p)
        base = _poly_mul_mod_fp(base, base, mod, p)
        e >>= 1
    return acc


def get_conway_polynomial(p: int, m: int) -> tuple[int, ...]:
    """Return the Conway polynomial for ``GF(p^m)`` or a deterministic primitive polynomial.

    Coefficients are returned in ascending degree order ``(c_0, c_1, ..., c_m)``
    with ``c_m == 1``.
    """
    key = (int(p), int(m))
    if key in CONWAY_POLYNOMIALS:
        return CONWAY_POLYNOMIALS[key]
    if key in _FALLBACK_CACHE:
        return _FALLBACK_CACHE[key]

    p_int, m_int = key
    if m_int == 1:
        g = find_primitive_root_mod_p(p_int)
        poly = ((-g) % p_int, 1)
        _FALLBACK_CACHE[key] = poly
        return poly

    q = p_int**m_int
    order = q - 1
    pf = prime_factors(order)
    one = [1] + [0] * (m_int - 1)

    # For binary fields, try sparse trinomials/pentanomials first for speed on large m
    if p_int == 2 and m_int > 16:
        for k1 in range(1, m_int):
            coeffs_list = [0] * (m_int + 1)
            coeffs_list[0] = 1
            coeffs_list[k1] = 1
            coeffs_list[m_int] = 1
            mod = tuple(coeffs_list)
            if _x_pow_mod_fp(order, mod, 2) == one and all(
                _x_pow_mod_fp(order // f, mod, 2) != one for f in pf
            ):
                _FALLBACK_CACHE[key] = mod
                return mod
        for k1 in range(1, m_int):
            for k2 in range(k1 + 1, m_int):
                for k3 in range(k2 + 1, m_int):
                    coeffs_list = [0] * (m_int + 1)
                    coeffs_list[0] = 1
                    coeffs_list[k1] = 1
                    coeffs_list[k2] = 1
                    coeffs_list[k3] = 1
                    coeffs_list[m_int] = 1
                    mod = tuple(coeffs_list)
                    if _x_pow_mod_fp(order, mod, 2) == one and all(
                        _x_pow_mod_fp(order // f, mod, 2) != one for f in pf
                    ):
                        _FALLBACK_CACHE[key] = mod
                        return mod

    # Deterministic lexicographic search over monic degree-m polynomials with c_0 != 0.
    # Having x of multiplicative order p^m - 1 modulo `mod` guarantees irreducibility
    # and primitivity over GF(p).
    for coeffs in itertools.product(range(p_int), repeat=m_int):
        if coeffs[0] == 0:
            continue
        mod = coeffs + (1,)
        if _x_pow_mod_fp(order, mod, p_int) != one:
            continue
        if all(_x_pow_mod_fp(order // f, mod, p_int) != one for f in pf):
            _FALLBACK_CACHE[key] = mod
            return mod

    raise RuntimeError(f"Could not find primitive irreducible polynomial for GF({p_int}^{m_int})")
