"""Shared NumPy numerical helpers for matrix representations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _digit(flat: int, index: int, dims: Sequence[int]) -> int:
    """Little-endian digit of subsystem ``index`` in flat basis state ``flat``."""
    stride = 1
    for q in range(index):
        stride *= dims[q]
    return (flat // stride) % dims[index]


def _strides(dims: Sequence[int]) -> list[int]:
    """Little-endian place values: ``stride[q] = prod(dims[:q])``."""
    strides = [1] * len(dims)
    for q in range(1, len(dims)):
        strides[q] = strides[q - 1] * dims[q - 1]
    return strides


def _contract_local(
    m: np.ndarray, tensor: np.ndarray, axes: Sequence[int], total: int, k: int
) -> np.ndarray:
    """Contract a local ``[out, in]`` operator into ``axes`` of a tensor.

    ``m`` has ``2k`` axes (k out, then k in); ``tensor`` has ``total`` axes. The
    k input axes of ``m`` are contracted with ``axes`` and the resulting out axes
    are permuted back into their positions. This is the single matrix-application
    core both apply kernels share: the statevector kernel calls it once on the
    n-axis state tensor, the density-matrix kernel twice (ket then bra) on the
    2n-axis rho tensor.
    """
    out = np.tensordot(m, tensor, axes=(list(range(k, 2 * k)), list(axes)))
    # Result axes: [out_0..out_{k-1}] + remaining tensor axes (original order).
    remaining = [ax for ax in range(total) if ax not in axes]
    perm = [0] * total
    for j, ax in enumerate(axes):
        perm[ax] = j
    for idx, ax in enumerate(remaining):
        perm[ax] = k + idx
    return np.transpose(out, perm)


def _measured_keep_mask(
    idx: int, subsystems: Sequence[int], dims: Sequence[int], size: int
) -> np.ndarray:
    """Boolean mask over flat basis states matching ``idx``'s measured digits.

    The projector core both collapse kernels share: an entry is kept when every
    measured subsystem's little-endian digit equals the corresponding digit of
    the sampled index ``idx``.
    """
    subsystems = list(subsystems)
    if len(set(subsystems)) == len(dims):
        keep = np.zeros(size, dtype=bool)
        keep[idx] = True
        return keep
    strides = _strides(dims)
    basis = np.arange(size)
    # One (N, m) broadcast instead of m separate O(N) passes: fold every measured
    # subsystem's stride and modulus into a single vectorized divide/mod/compare.
    stride_arr = np.array([strides[q] for q in subsystems])
    dim_arr = np.array([dims[q] for q in subsystems])
    digits = (basis[:, None] // stride_arr) % dim_arr
    idx_digits = (idx // stride_arr) % dim_arr
    return np.all(digits == idx_digits, axis=1)
