"""Python-callable Numba matrix operations over explicit native state.

Engines own cache lifetimes and execution policy. These functions use those
numerical resources without retaining an engine or an evolving state. Target
layouts are complete before execution and are only read by these operations.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from ....._backends.steps import ApplyChannelStep, ApplyMatrixStep
from ._apply import _apply_matrix, _classify_matrix
from .....noise.nb import _jump_branch_kernel
from ..numpy_kernels._state import _ChannelRouteCache, _sample_unitary
from ._matrix_cache import (
    _MatrixStructureCache,
    _SuperoperatorCache,
)


def _apply_matrix_statevector(
    state: np.ndarray,
    step: ApplyMatrixStep,
    dims: Sequence[int],
    *,
    structures: _MatrixStructureCache,
    plans: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Apply one gate using cached classification and target layout."""
    code, columns, values = structures.resolve(step)
    targets = tuple(step.target_indices)
    plan = plans[targets]
    return _apply_matrix(state, step.matrix, plan, code, columns, values)


def _apply_superoperator_to_state(
    state: np.ndarray,
    step: ApplyMatrixStep | ApplyChannelStep,
    dims: Sequence[int],
    *,
    superoperators: _SuperoperatorCache,
    plans: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Apply a cached local superoperator without copying contiguous state.

    Gates and exact Kraus sums share this numerical operation. A sparse local
    superoperator uses CSR arithmetic; the evolving state remains dense.
    """
    superop, code, columns, values, sparse = superoperators.resolve(step)
    targets = tuple(step.target_indices)
    plan = plans[targets]
    return _apply_matrix(state, superop, plan, code, columns, values, sparse=sparse)


def _apply_unclassified_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    *,
    plans: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Classify a bare Kraus matrix once and use the shared matrix launcher."""
    matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
    columns = np.empty(matrix.shape[0], dtype=np.int64)
    values = np.empty(matrix.shape[0], dtype=np.complex128)
    code = int(_classify_matrix(matrix, columns, values))
    targets = tuple(targets)
    plan = plans[targets]
    return _apply_matrix(state, matrix, plan, code, columns, values)


def _apply_channel_statevector(
    state: np.ndarray,
    step: ApplyChannelStep,
    dims: Sequence[int],
    rng: np.random.Generator,
    *,
    routes: _ChannelRouteCache,
    plans: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Sample a fixed unitary branch or a state-dependent Kraus branch.

    General branches use separate copies because the kernels mutate their input.
    The compiled branch sampler retains the existing weights and draw sequence.
    """
    # pylint: disable-next=fixme
    # TODO: Benchmark per-channel-step caching of branch matrix classification
    # (including sampled-unitary matrices) before adopting it. Verify seeded
    # results and cache lifetimes; keep state-dependent Kraus probabilities per shot.
    route = routes.resolve(step)
    if route is not None:
        matrix = _sample_unitary(route, rng)
        if matrix is None:
            return state
        return _apply_unclassified_matrix(
            state, matrix, step.target_indices, plans=plans
        )
    branches = np.empty((len(step.kraus_ops), state.shape[0]), dtype=np.complex128)
    for i, kraus in enumerate(step.kraus_ops):
        branches[i] = _apply_unclassified_matrix(
            state.copy(), kraus, step.target_indices, plans=plans
        )
    return _jump_branch_kernel(branches, float(rng.random()))


def _apply_channel_density_matrix(
    state: np.ndarray,
    step: ApplyChannelStep,
    dims: Sequence[int],
    rng: np.random.Generator,
    *,
    superoperators: _SuperoperatorCache,
    plans: Mapping[tuple[int, ...], tuple],
) -> np.ndarray:
    """Apply the exact channel superoperator without consuming randomness."""
    return _apply_superoperator_to_state(
        state, step, dims, superoperators=superoperators, plans=plans
    )
