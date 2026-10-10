"""Statevector channel arithmetic for compiled Numba trajectories.

The caller selects a channel-table entry and supplies one uniform draw.
Conditions, draw cursors, and shot sequencing remain in the workflow.
Branch weighing preserves each engine's seeded behavior but need not produce
bit-identical per-seed counts to the NumPy trajectory implementation.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from numba import njit

from .....noise.nb import _inverse_cdf_pick
from ._apply import (
    _advance_base,
    _apply_resolved_range,
    _classify_matrix,
    _spread_base,
)


@njit(cache=True)
def _reduced_density(
    state, offsets, comp_strides, comp_dims, cosets
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Reduced density matrix ``rho_T`` of the target subsystems, ``d x d``.

    Traces the flat state over the complement (non-target) subsystems:
    ``rho_T[a, b] = sum_cosets psi[base + offsets[a]] * conj(psi[base +
    offsets[b]])``, one pass over every amplitude with the coset kernels'
    division-free odometer. The ``O(size * d)`` object that lets a channel weigh
    all ``num`` Kraus branches without applying any (see `_apply_channel_step`).
    """
    d = offsets.shape[0]
    rho = np.zeros((d, d), dtype=np.complex128)
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    for _ in range(cosets):
        for a in range(d):
            amplitude = state[base + offsets[a]]
            for b in range(d):
                rho[a, b] += amplitude * state[base + offsets[b]].conjugate()
        base = _advance_base(base, comp_strides, comp_dims, counter)
    return rho


@njit(cache=True)
def _reduced_diagonal(
    state, offsets, comp_strides, comp_dims, cosets
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Diagonal of the target subsystems' reduced density matrix, ``d`` reals.

    ``rho_T[a, a] = sum_cosets |psi[base + offsets[a]]|**2`` - the marginal
    probability of each local basis state. One pass over the state, where
    `_reduced_density` does ``d`` accumulations per amplitude to build all
    ``d**2`` entries. Enough whenever every ``M_i`` is diagonal, since then
    ``Tr(M_i rho_T) = sum_a M_i[a, a] rho_T[a, a]``.
    """
    d = offsets.shape[0]
    diagonal = np.zeros(d, dtype=np.float64)
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(0, comp_strides, comp_dims, counter)
    for _ in range(cosets):
        for a in range(d):
            amplitude = state[base + offsets[a]]
            diagonal[a] += (
                amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
            )
        base = _advance_base(base, comp_strides, comp_dims, counter)
    return diagonal


@njit(cache=True)
def _apply_channel_step(
    state,
    c,
    ch_kra_ptr,
    ch_num_kraus,
    ch_dim,
    ch_off_ptr,
    ch_comp_ptr,
    ch_comp_len,
    ch_cdf_ptr,
    ch_mmat_diag,
    ch_kra_flat,
    ch_mmat_flat,
    ch_off_flat,
    ch_comp_stride_flat,
    ch_comp_dim_flat,
    ch_cdf_flat,
    ch_ident_flat,
    size,
    u,
) -> None:  # pragma: no cover - compiled by Numba
    """Sample one Kraus branch of compiled channel ``c`` and apply it in place.

    Which of two weighings runs is decided once per run by
    `_compile_channel_table`:

    - ``ch_cdf_ptr[c] >= 0`` - **Pauli sampling.** The operators are scaled
      unitaries, so ``p_i = <psi|K_i^dagger K_i|psi>`` does not depend on the
      state and its cdf is precomputed. No amplitude is read to make the draw,
      and a draw landing on the identity - the dominant branch at any physical
      error rate - returns having touched nothing.
    - otherwise - **quantum-jump weighing**: ``p_i`` equals ``Tr(M_i rho_T)``
      over the targets' reduced density matrix ``rho_T`` (`_reduced_density`;
      ``M_i = K_i^dagger K_i`` precomputed in the channel table), which is
      ``O(size * d + num * d**2)`` against materializing and norming ``num``
      full branches. ``ch_mmat_diag[c]`` narrows that to the ``d`` times
      cheaper `_reduced_diagonal` when every ``M_i`` is diagonal.

    Either way the drawn operator is applied in place at unit norm: the
    Pauli-sampled operators arrive pre-scaled, and a jump branch is divided by
    the ``sqrt(p_chosen)`` the weighing already computed - both on the ``d x d``
    operator, so nothing passes over the state to renormalize. The round-off
    that leaves in the norm is absorbed by `_sv_probabilities_kernel` and
    `_sv_project_kernel`.

    Both consume exactly one uniform ``u``, so which weighing a channel takes
    never shifts the shot's RNG stream.

    A mathematically equal but numerically distinct estimator of the same
    channel as the NumPy reference: distributions agree and each engine stays
    seed-reproducible, but per-seed counts are not bit-identical across the two
    (see the module docstring's contract).
    """
    d = ch_dim[c]
    num = ch_num_kraus[c]
    off_start = ch_off_ptr[c]
    offsets = ch_off_flat[off_start : off_start + d]
    comp_start = ch_comp_ptr[c]
    comp_end = comp_start + ch_comp_len[c]
    comp_strides = ch_comp_stride_flat[comp_start:comp_end]
    comp_dims = ch_comp_dim_flat[comp_start:comp_end]
    kra_start = ch_kra_ptr[c]
    cosets = size // d

    cdf_start = ch_cdf_ptr[c]
    if cdf_start >= 0:
        # Probing the leading entry first is a shortcut, not a special case: it
        # is what a search returns for any u below it, and the dominant branch.
        if u < ch_cdf_flat[cdf_start]:
            chosen = 0
        else:
            lo = 1
            hi = num
            while lo < hi:
                mid = (lo + hi) // 2
                if ch_cdf_flat[cdf_start + mid] <= u:
                    lo = mid + 1
                else:
                    hi = mid
            chosen = lo if lo < num else num - 1
        if ch_ident_flat[cdf_start + chosen] != 0:
            return
        scale = 1.0
    else:
        weights = np.empty(num, dtype=np.float64)
        if ch_mmat_diag[c] != 0:
            diagonal = _reduced_diagonal(
                state, offsets, comp_strides, comp_dims, cosets
            )
            for i in range(num):
                mmat_start = kra_start + i * d * d
                trace = 0.0
                for a in range(d):
                    trace += ch_mmat_flat[mmat_start + a * d + a].real * diagonal[a]
                # Tr(M_i rho_T) is real and PSD-nonnegative; clamp round-off.
                weights[i] = trace if trace > 0.0 else 0.0
        else:
            rho = _reduced_density(state, offsets, comp_strides, comp_dims, cosets)
            for i in range(num):
                mmat_start = kra_start + i * d * d
                mmat = ch_mmat_flat[mmat_start : mmat_start + d * d].reshape(d, d)
                trace = 0.0
                for a in range(d):
                    for b in range(d):
                        trace += (mmat[a, b] * rho[b, a]).real
                weights[i] = trace if trace > 0.0 else 0.0
        chosen = _inverse_cdf_pick(weights, u)
        # The drawn branch's squared norm is its own weight. A zero-weight
        # branch has zero cdf width and can never be drawn.
        scale = 1.0 / sqrt(weights[chosen])

    kraus_start = chosen * d * d + kra_start
    operator = np.empty((d, d), dtype=np.complex128)
    for a in range(d):
        for b in range(d):
            operator[a, b] = ch_kra_flat[kraus_start + a * d + b] * scale
    columns = np.empty(d, dtype=np.int64)
    values = np.empty(d, dtype=np.complex128)
    code = _classify_matrix(operator, columns, values)
    _apply_resolved_range(
        state,
        code,
        operator,
        columns,
        values,
        offsets,
        comp_strides,
        comp_dims,
        0,
        cosets,
    )
