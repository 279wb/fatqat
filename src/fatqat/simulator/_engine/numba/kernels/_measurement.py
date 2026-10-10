"""Physical measurement and reset-shift primitives for Numba states."""

from __future__ import annotations

from math import sqrt

import numpy as np
from numba import njit

# --- shared sampling ---


@njit(cache=True)
def _normalized_cdf(
    probabilities: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Cumulative distribution normalized by its last element (``cdf /= cdf[-1]``).

    Built with the same sequential cumulative sum NumPy's ``cumsum`` uses, so a
    ``searchsorted`` against it matches ``rng.choice``'s internal inverse-CDF.
    """
    size = probabilities.shape[0]
    cdf = np.empty(size, dtype=np.float64)
    running = 0.0
    for i in range(size):
        running += probabilities[i]
        cdf[i] = running
    last = cdf[size - 1]
    for i in range(size):
        cdf[i] = cdf[i] / last
    return cdf


@njit(cache=True)
def _searchsorted_right(
    cdf: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Index of the first ``cdf`` entry strictly greater than ``u``.

    Equivalent to ``cdf.searchsorted(u, side="right")`` for a non-decreasing
    ``cdf`` - the inverse-CDF step of categorical sampling.
    """
    lo = 0
    hi = cdf.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] <= u:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _sample_index_kernel(
    probabilities: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Draw one flat basis index from ``probabilities`` given uniform ``u``."""
    return _searchsorted_right(_normalized_cdf(probabilities), u)


@njit(cache=True)
def _sample_indices_kernel(
    probabilities: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Draw one flat index per uniform in ``uniforms`` (built cdf reused)."""
    cdf = _normalized_cdf(probabilities)
    out = np.empty(uniforms.shape[0], dtype=np.int64)
    for j in range(uniforms.shape[0]):
        out[j] = _searchsorted_right(cdf, uniforms[j])
    return out


# --- statevector ---


@njit(cache=True)
def _sv_probabilities_kernel(
    state: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Computational-basis probabilities of a flat statevector.

    Mirrors ``np.abs(state) ** 2`` normalized by its sum: ``abs`` of a complex
    value is ``sqrt(re^2 + im^2)``, squared back to the Born probability.
    """
    size = state.shape[0]
    probabilities = np.empty(size, dtype=np.float64)
    total = 0.0
    for i in range(size):
        magnitude = abs(state[i])
        probabilities[i] = magnitude * magnitude
        total += probabilities[i]
    if total > 0.0:
        for i in range(size):
            probabilities[i] = probabilities[i] / total
    return probabilities


@njit(cache=True)
def _sv_project_kernel(
    state: np.ndarray,
    measured_strides: np.ndarray,
    measured_dims: np.ndarray,
    index: int,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Project onto the measured digits of ``index`` and renormalize.

    Keeps every basis state whose measured subsystem digits match those of the
    sampled ``index``, zeros the rest, then divides by the surviving norm -
    exactly ``np.linalg.norm``-normalized projective collapse.
    """
    size = state.shape[0]
    m = measured_strides.shape[0]
    index_digits = np.empty(m, dtype=np.int64)
    for j in range(m):
        index_digits[j] = (index // measured_strides[j]) % measured_dims[j]

    out = np.zeros(size, dtype=np.complex128)
    norm_sq = 0.0
    for flat in range(size):
        keep = True
        for j in range(m):
            if (flat // measured_strides[j]) % measured_dims[j] != index_digits[j]:
                keep = False
                break
        if keep:
            amplitude = state[flat]
            out[flat] = amplitude
            norm_sq += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    if norm_sq > 0.0:
        norm = sqrt(norm_sq)
        for flat in range(size):
            out[flat] = out[flat] / norm
    return out


@njit(cache=True)
def _sv_shift_subsystem(
    state: np.ndarray, stride: int, dim: int, outcome: int
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Cyclically shift one subsystem's digit down by ``outcome`` (reset to |0>).

    Equivalent to applying ``shift_matrix(dim, -outcome)`` to the subsystem: the
    amplitude with subsystem digit ``g`` moves to digit ``(g - outcome) mod dim``.
    After a collapse only digit ``outcome`` is populated, so this relabels the
    measured branch to |0>.
    """
    size = state.shape[0]
    out = np.zeros(size, dtype=np.complex128)
    for flat in range(size):
        digit = (flat // stride) % dim
        shifted = (digit - outcome + dim) % dim
        out[flat + (shifted - digit) * stride] = state[flat]
    return out


# --- density matrix ---


@njit(cache=True)
def _dm_probabilities_kernel(
    rho: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Computational-basis probabilities from the diagonal of ``rho``.

    Mirrors ``clip(real(diag(rho)), 0, None)`` normalized by its sum: a valid
    density matrix has a real, non-negative diagonal, so tiny negative round-off
    is clamped before normalizing into a sampling distribution.
    """
    size = rho.shape[0]
    probabilities = np.empty(size, dtype=np.float64)
    total = 0.0
    for i in range(size):
        value = rho[i, i].real
        value = max(value, 0.0)
        probabilities[i] = value
        total += value
    if total > 0.0:
        for i in range(size):
            probabilities[i] = probabilities[i] / total
    return probabilities


@njit(cache=True)
def _dm_project_kernel(
    rho: np.ndarray,
    measured_strides: np.ndarray,
    measured_dims: np.ndarray,
    index: int,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Project ``rho`` onto the measured digits of ``index`` and renormalize.

    Keeps the block of basis states whose measured subsystem digits match those
    of the sampled ``index`` (both row and column must match), zeros the rest,
    then divides by the surviving trace - exactly the trace-normalized
    projective collapse ``rho * keep[:, None] * keep[None, :] / Tr``.
    """
    size = rho.shape[0]
    m = measured_strides.shape[0]
    index_digits = np.empty(m, dtype=np.int64)
    for j in range(m):
        index_digits[j] = (index // measured_strides[j]) % measured_dims[j]

    keep = np.empty(size, dtype=np.bool_)
    for i in range(size):
        match = True
        for j in range(m):
            if (i // measured_strides[j]) % measured_dims[j] != index_digits[j]:
                match = False
                break
        keep[i] = match

    trace = 0.0
    for i in range(size):
        if keep[i]:
            trace += rho[i, i].real
    # Mirror ``new / trace if trace > 0 else new``: on a zero-trace branch leave
    # the (already all-zero) kept block unscaled instead of dividing by zero.
    scale = 1.0 / trace if trace > 0.0 else 1.0

    out = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        if keep[i]:
            for j in range(size):
                if keep[j]:
                    out[i, j] = rho[i, j] * scale
    return out
