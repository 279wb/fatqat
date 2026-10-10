"""Host-side numerical operand preparation and matrix caches for Numba.

Statevectors use physical subsystem dimensions; density matrices reuse the same
layouts over doubled bra/ket dimensions. These helpers prepare numerical operands
and kernel chunk counts without owning workflow sequencing. Matrix resolutions
are retained across trajectories and pruned when the effective plan changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np
from numba import config as numba_config

from ....._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    BuiltinKernelKey,
    ResolvedStep,
)
from .....noise.nb import _kraus_stack, _kraus_superop_kernel
from ._apply import _DENSE, _DIAGONAL, _PERMUTATION, _classify_matrix

# Declared structure per canonical gate identity: the key-to-kernel table.
# The numerical runtime owns these declarations; the key carries only identity.
#
# How kernel selection flows (one classifier, one kernel family):
#   standard path   apply(step) -> _MatrixStructureCache.resolve(step): a key declared
#                   _DENSE skips the scan entirely; everything else takes ONE
#                   `_classify_matrix` scan at plan preparation, cached per
#                   step - never per application, never per shot.
#   fallback path   _apply_local(state, matrix, targets): matrix-only callers
#                   (noise Kraus branches) - the same single
#                   `_classify_matrix` scan, then the same resolved kernels.
#   compiled path   _compile_dynamic_plan bakes each gate's resolved code into
#                   the plan arrays, so the shots kernel never classifies.
#
# An entry may claim _DIAGONAL or _PERMUTATION only if that structure holds
# for EVERY parameter value of the gate (RZ is diagonal for all theta); a
# parametric gate whose structure varies claims _DENSE, which is always safe
# (RX at theta=pi happens to be a permutation, but _DENSE stays correct).
# The spec-vs-content test in test_kernel_dispatch.py enforces agreement
# with `_classify_matrix` over the whole default map.
_K = BuiltinKernelKey
_KERNEL_SPECS: dict[BuiltinKernelKey, int] = {
    # diagonal for every parameter value
    _K.I: _DIAGONAL,
    _K.Z: _DIAGONAL,
    _K.S: _DIAGONAL,
    _K.SDG: _DIAGONAL,
    _K.T: _DIAGONAL,
    _K.TDG: _DIAGONAL,
    _K.CZ: _DIAGONAL,
    _K.CS: _DIAGONAL,
    _K.RZ: _DIAGONAL,
    _K.PHASE: _DIAGONAL,
    _K.U1: _DIAGONAL,
    _K.CPHASE: _DIAGONAL,
    _K.CLOCK: _DIAGONAL,
    _K.CCLOCK: _DIAGONAL,
    _K.SUBSPACE_RZ: _DIAGONAL,
    # exactly one nonzero per row, for every parameter value
    _K.X: _PERMUTATION,
    _K.Y: _PERMUTATION,
    _K.CX: _PERMUTATION,
    _K.CY: _PERMUTATION,
    _K.SWAP: _PERMUTATION,
    _K.ISWAP: _PERMUTATION,
    _K.CCX: _PERMUTATION,
    _K.CSWAP: _PERMUTATION,
    _K.SHIFT: _PERMUTATION,
    _K.SUM: _PERMUTATION,
    _K.SWAP_LEVELS: _PERMUTATION,
    # generically dense
    _K.H: _DENSE,
    _K.SX: _DENSE,
    _K.RX: _DENSE,
    _K.RY: _DENSE,
    _K.U: _DENSE,
    _K.U2: _DENSE,
    _K.U3: _DENSE,
    _K.FOURIER: _DENSE,
    _K.FOURIERDG: _DENSE,
    _K.SUBSPACE_RX: _DENSE,
    _K.SUBSPACE_RY: _DENSE,
}


class _MatrixStructureCache:
    """Resolve frozen gate matrices independently of workflow packing.

    Per-step resolutions (code/columns/values) are keyed by ``id(step)`` and
    pin the source step so a recycled identity can never alias another entry.
    Structure depends only on the step's frozen matrix, not system dimensions.
    Trajectory initialization deliberately keeps these entries: the dynamic
    workflow initializes each shot and must reuse its once-per-plan resolutions.
    Materializing the next plan prunes steps it no longer uses through ``retain``.
    """

    def __init__(self):
        self._entries: dict[int, tuple] = {}

    def retain(self, plan: tuple[ResolvedStep, ...]) -> None:
        """Keep only resolutions used by the next effective plan."""
        active_ids = {id(step) for step in plan}
        self._entries = {
            step_id: entry
            for step_id, entry in self._entries.items()
            if step_id in active_ids
        }

    def resolve(self, step: ApplyMatrixStep) -> tuple[int, np.ndarray, np.ndarray]:
        """Resolve a step's kernel structure once: the standard-path front end.

        `_classify_matrix` is the single classification rule. A kernel_key
        declared ``_DENSE`` skips the scan entirely (dense handles any matrix
        and its columns/values are never read); every other step - declared
        diagonal/permutation or un-keyed - takes one scan here, at plan
        preparation, cached per step (id-keyed, identity-pinned) instead of
        once per application. For declared codes the scan reproduces the
        declaration (guaranteed for every parameter value by the
        spec-vs-content test); content is always the executable truth.
        """
        cached = self._entries.get(id(step))
        if cached is not None and cached[0] is step:
            return cached[1]
        matrix = np.ascontiguousarray(step.matrix, dtype=np.complex128)
        d = matrix.shape[0]
        columns = np.empty(d, dtype=np.int64)
        values = np.empty(d, dtype=np.complex128)
        if _KERNEL_SPECS.get(step.kernel_key) == _DENSE:
            code = _DENSE  # declared dense: nothing to extract, no scan
        else:
            code = int(_classify_matrix(matrix, columns, values))
        resolved = (code, columns, values)
        self._entries[id(step)] = (step, resolved)
        return resolved


class _SuperoperatorCache:
    """Prepare gate/channel superoperators independently of workflow packing.

    Per-step resolutions (matrix/code/columns/values/sparse) are keyed by
    ``id(step)`` and pin the source step so recycled identities cannot alias.
    Resolutions depend only on frozen gate/channel payloads, not system dimensions.
    Trajectory initialization deliberately keeps them so each shot reuses the
    once-per-plan construction and classification. Materializing the next plan
    prunes steps it no longer uses through ``retain``.
    """

    def __init__(self):
        self._entries: dict[int, tuple] = {}

    def retain(self, plan: tuple[ResolvedStep, ...]) -> None:
        """Keep only resolutions used by the next effective plan."""
        active_ids = {id(step) for step in plan}
        self._entries = {
            step_id: entry
            for step_id, entry in self._entries.items()
            if step_id in active_ids
        }

    def resolve(
        self, step: ApplyMatrixStep | ApplyChannelStep
    ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, tuple | None]:
        """Build and classify a step's super-operator once per plan.

        For a gate the super-operator is ``kron(M, conj(M))``; the Kronecker
        product preserves structure, so a kernel_key declared ``_DENSE`` skips
        the scan and every other gate takes one `_classify_matrix` scan of the
        super-operator. A channel's ``sum_i kron(K_i, conj(K_i))``
        (`noise.nb._kraus_superop_kernel`) carries no key and is always
        scanned - which is what lets a diagonal channel (phase damping) reach
        the diagonal kernel.

        A dense-classified but mostly-zero super-operator (e.g. depolarizing)
        additionally gets a CSR (`_superop_csr`) so the pass can skip its zeros;
        the returned ``sparse`` is that CSR or ``None``. Cached per step
        (id-keyed, identity-pinned), resolved once per plan, not per trajectory.
        """
        cached = self._entries.get(id(step))
        if cached is not None and cached[0] is step:
            return cached[1]
        if isinstance(step, ApplyMatrixStep):
            m = np.ascontiguousarray(step.matrix, dtype=np.complex128)
            superop = np.kron(m, m.conj())
            declared_dense = _KERNEL_SPECS.get(step.kernel_key) == _DENSE
        else:
            superop = _kraus_superop_kernel(_kraus_stack(step.kraus_ops))
            declared_dense = False
        d = superop.shape[0]
        columns = np.empty(d, dtype=np.int64)
        values = np.empty(d, dtype=np.complex128)
        if declared_dense:
            code = _DENSE  # declared dense: nothing to extract, no scan
        else:
            code = int(_classify_matrix(superop, columns, values))
        sparse = _superop_csr(superop) if code == _DENSE else None
        resolved = (superop, code, columns, values, sparse)
        self._entries[id(step)] = (step, resolved)
        return resolved


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
