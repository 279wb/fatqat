"""Host-side numerical layout and sparse-matrix preparation for Numba.

Statevectors use physical subsystem dimensions; density matrices reuse the same
layouts over doubled bra/ket dimensions. These helpers prepare numerical operands
and kernel chunk counts without owning workflow sequencing or engine caches.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np
from numba import config as numba_config

# Numba fixes the launch pool's capacity at import time. ``set_num_threads``
# changes only the active mask, so policy ceilings clamp to this configured
# capacity rather than to whichever smaller mask the caller currently uses.
_MAX_THREADS = int(numba_config.NUMBA_NUM_THREADS)
# A coset walk goes parallel only once each worker thread would get at least
# this many amplitudes of work; below that the parallel-region launch/sync cost
# outweighs the memory-bandwidth-bound work it saves. Expressed per-thread so
# the absolute floor scales with the core count (the dominant machine variable)
# as ``_MAX_THREADS * grain``, not a fixed size. The grains reproduce the
# previously hand-tuned 2^15 / 2^18 floors at ~16 threads; the density matrix's
# 4^n pass is more bandwidth-bound, so it takes a larger grain. The floor
# affects speed only - chunking is bit-identical - so a coarse grain suffices.
_GRAIN_TO_THREAD = 1 << 11  # statevector: min amplitudes per thread
_GRAIN_TO_THREAD_DM = 1 << 14  # density matrix: min amplitudes per thread
_MIN_SIZE_TO_THREAD = _MAX_THREADS * _GRAIN_TO_THREAD
_MIN_SIZE_TO_THREAD_DM = _MAX_THREADS * _GRAIN_TO_THREAD_DM


def _plan_chunks(
    num_cosets: int,
    size: int,
    min_parallel_size: int = _MIN_SIZE_TO_THREAD,
    *,
    max_workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> int:
    """Parallel chunk count for a gate on a ``size``-amplitude state.

    Returns 1 (the serial kernel, no parallel-region overhead) for states below
    ``min_parallel_size``; otherwise splits the cosets across all worker threads.
    """
    if max_workers <= 1:
        return 1
    if not force_parallel and size < min_parallel_size:
        return 1
    return max(1, min(max_workers, num_cosets))


def _compute_apply_plan(
    dims: Sequence[int],
    targets: Sequence[int],
    min_parallel_size: int = _MIN_SIZE_TO_THREAD,
    *,
    max_workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> tuple:
    """Precompute the strided-block kernel layout for a target tuple.

    Returns ``offsets[c]`` (the flat offset of local index ``c``, with
    ``targets[0]`` most-significant), the flat strides/dimensions of the
    complement (non-target) subsystems the kernel odometers over, the coset
    count, and the parallel chunk count for this many cosets.

    Depends only on ``dims`` and ``targets``, so the statevector path calls it
    with the physical system dims while the density-matrix path calls it with
    the doubled ``bra + ket`` dims (see the module docstring).
    """
    n = len(dims)
    local_dims = [dims[t] for t in targets]
    local_dim = prod(local_dims)

    strides = [prod(dims[:q]) for q in range(n)]
    target_strides = [strides[t] for t in targets]
    # Mixed-radix place values with targets[0] most-significant.
    local_places = [1] * len(targets)
    for j in range(len(targets) - 2, -1, -1):
        local_places[j] = local_places[j + 1] * local_dims[j + 1]

    local_index = np.arange(local_dim, dtype=np.int64)
    offsets = np.zeros(local_dim, dtype=np.int64)
    for j, stride in enumerate(target_strides):
        offsets += ((local_index // local_places[j]) % local_dims[j]) * stride

    target_set = set(targets)
    complement = [q for q in range(n) if q not in target_set]
    comp_strides = np.array([strides[q] for q in complement], dtype=np.int64)
    comp_dims = np.array([dims[q] for q in complement], dtype=np.int64)
    size = prod(dims)
    num_cosets = size // local_dim
    n_chunks = _plan_chunks(
        num_cosets,
        size,
        min_parallel_size,
        max_workers=max_workers,
        force_parallel=force_parallel,
    )
    return offsets, comp_strides, comp_dims, num_cosets, n_chunks


def _measured_layout(
    dims: Sequence[int], subsystems: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Flat strides and dimensions of the measured subsystems for the kernels."""
    strides = np.array([prod(dims[:q]) for q in subsystems], dtype=np.int64)
    measured_dims = np.array([dims[q] for q in subsystems], dtype=np.int64)
    return strides, measured_dims


# Walk a super-operator sparsely only when its nonzero fraction is below this,
# where skipping zeros beats the CSR indirection (two-qubit depolarizing sits
# near 0.11, one-qubit near 0.375; a dense gate super-operator is 1.0).
_SPARSE_SUPEROP_FRACTION = 0.5


def _superop_csr(matrix: np.ndarray) -> tuple | None:
    """CSR of ``matrix``'s exact nonzeros, or ``None`` if it is too dense.

    Lists each row's nonzeros in increasing column order, so a sparse walk
    reproduces the dense dot bit for bit - a skipped entry is an exact zero the
    dense sum would have added as ``0.0``, in the same column order. Returns
    ``(indptr, indices, data)``; ``None`` when the nonzero fraction is above
    `_SPARSE_SUPEROP_FRACTION`, where the indirection would not pay off.
    """
    # CSR stores only the local gate/channel super-operator. The evolving
    # density matrix or accumulated full-system super-operator stays dense;
    # these arrays let application kernels skip zeros in the local operation.
    d = matrix.shape[0]
    indptr = np.empty(d + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[complex] = []
    indptr[0] = 0
    for r in range(d):
        row = matrix[r]
        for c in range(d):
            if row[c] != 0.0:
                indices.append(c)
                data.append(complex(row[c]))
        indptr[r + 1] = len(indices)
    if len(indices) > _SPARSE_SUPEROP_FRACTION * d * d:
        return None
    return (
        indptr,
        np.asarray(indices, dtype=np.int64),
        np.asarray(data, dtype=np.complex128),
    )
