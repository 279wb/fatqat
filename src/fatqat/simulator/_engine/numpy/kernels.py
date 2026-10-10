"""Shared NumPy numerical helpers for matrix representations."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from ...._backends.steps import ApplyChannelStep, ResolvedStep
from ....noise.base import _sampled_unitary_branches

# What `_sampled_unitary_branches` resolves a channel step to: branch
# probabilities, unit-norm operators, identity flags - or None for a channel
# that must be weighed against the state.
_Branches = tuple[np.ndarray, tuple[np.ndarray, ...], tuple[bool, ...]] | None


class _ChannelRouteCache:
    """Prepare state-independent channel branches across shot initialization.

    Entries pin their source steps to prevent recycled identities from aliasing.
    Routes depend only on frozen Kraus operators, not dimensions or live state.
    Materializing a new plan retains shared steps and releases replaced ones.
    """

    def __init__(self):
        self._entries: dict[int, tuple[ApplyChannelStep, _Branches]] = {}

    def retain(self, plan: tuple[ResolvedStep, ...]) -> None:
        """Keep only resolutions used by the next effective plan."""
        active_ids = {id(step) for step in plan}
        self._entries = {
            step_id: entry
            for step_id, entry in self._entries.items()
            if step_id in active_ids
        }

    def resolve(self, step: ApplyChannelStep) -> _Branches:
        """Resolve a channel step's branch-weighing route once, then cache it."""
        cached = self._entries.get(id(step))
        if cached is not None and cached[0] is step:
            return cached[1]
        branches = _sampled_unitary_branches(step.kraus_ops)
        self._entries[id(step)] = (step, branches)
        return branches


# This cache has one operation: resolve an immutable reset channel on demand.
class _ResetChannelCache:  # pylint: disable=too-few-public-methods
    """Reuse reset channels by subsystem and dimension across executions."""

    def __init__(self):
        self._entries: dict[tuple[int, int], ApplyChannelStep] = {}

    def resolve(self, index: int, dim: int) -> ApplyChannelStep:
        """Return the single-subsystem reset channel, building it once."""
        step = self._entries.get((index, dim))
        if step is None:
            kraus = []
            for outcome in range(dim):
                operator = np.zeros((dim, dim), dtype=complex)
                operator[0, outcome] = 1.0
                kraus.append(operator)
            step = ApplyChannelStep(kraus_ops=tuple(kraus), target_indices=(index,))
            self._entries[(index, dim)] = step
        return step


def _allocate_statevector(size: int, initial_state: np.ndarray | None) -> np.ndarray:
    """Return an owned ket, copied from input or in the computational zero state."""
    if initial_state is not None:
        # Copy: each shot evolves its own buffer, and the caller's array
        # must survive the run unchanged.
        return np.array(initial_state, dtype=complex).reshape(size)
    state = np.zeros(size, dtype=complex)
    state[0] = 1.0
    return state


def _allocate_density_matrix(size: int, initial_state: np.ndarray | None) -> np.ndarray:
    """Return an owned density matrix, accepting a ket as a pure state."""
    given = initial_state
    if given is not None:
        given = np.asarray(given, dtype=complex)
        # A ket is accepted as shorthand for the pure state it describes,
        # so the same array can start a statevector and a density-matrix
        # run without the caller forming the outer product twice.
        if given.ndim == 1:
            return np.outer(given, given.conj())
        return np.array(given, dtype=complex)
    state = np.zeros((size, size), dtype=complex)
    state[0, 0] = 1.0
    return state


def _allocate_unitary(size: int, initial_state: np.ndarray | None) -> np.ndarray:
    """Return the identity map over statevector columns."""
    assert initial_state is None, "operator execution has no initial state"
    return np.eye(size, dtype=complex)


def _allocate_superoperator(size: int, initial_state: np.ndarray | None) -> np.ndarray:
    """Return the identity map over density-matrix columns."""
    assert initial_state is None, "operator execution has no initial state"
    return np.eye(size * size, dtype=complex)


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


def _apply_statevector(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Apply a local matrix using the caller's prepared subsystem dimensions."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    psi = state.reshape(reversed_dims)
    target_axes = [n - 1 - q for q in targets]
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    return _contract_local(m, psi, target_axes, n, k).reshape(-1)


def _statevector_probabilities(state: np.ndarray) -> np.ndarray:
    """Return normalized basis probabilities, preserving an all-zero input."""
    probabilities = np.abs(state) ** 2
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _project_statevector(
    state: np.ndarray,
    idx: int,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
) -> np.ndarray:
    """Return a normalized physical projection without modifying the input."""
    keep = _measured_keep_mask(idx, measured_subsystems, dims, len(state))
    new = state.copy()
    new[~keep] = 0.0
    norm = np.linalg.norm(new)
    return new / norm if norm > 0 else new


def _apply_density_matrix(
    rho: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Return the local sandwich M rho M-adjoint using prepared dimensions."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    tensor = rho.reshape(reversed_dims * 2)
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    tensor = _contract_local(m, tensor, ket_axes, 2 * n, k)
    tensor = _contract_local(m.conj(), tensor, bra_axes, 2 * n, k)
    return tensor.reshape(rho.shape)


def _density_matrix_probabilities(rho: np.ndarray) -> np.ndarray:
    """Return normalized diagonal probabilities with round-off clipped."""
    # A valid rho's diagonal is real and non-negative; clip tiny round-off so
    # the values stay a valid sampling distribution.
    probabilities = np.clip(np.real(np.diagonal(rho)), 0.0, None)
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _project_density_matrix(
    rho: np.ndarray,
    idx: int,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
) -> np.ndarray:
    """Return a normalized physical projection without modifying the input."""
    keep = _measured_keep_mask(idx, measured_subsystems, dims, rho.shape[0])
    new = rho * keep[:, None] * keep[None, :]
    trace = np.real(np.trace(new))
    return new / trace if trace > 0 else new


def _reset_density_matrix(
    rho: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Trace out nonempty targets and reprepare their joint zero state."""
    n = len(dims)
    targets = tuple(indices)
    local_dims = tuple(dims[t] for t in targets)
    dt = prod(local_dims)
    rest = rho.shape[0] // dt

    tensor = rho.reshape(reversed_dims * 2)
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    moved = ket_axes + bra_axes
    # `remaining` stays ascending, so rest-ket axes precede rest-bra axes and
    # the (dt, dt, rest, rest) regroup below is valid.
    remaining = [ax for ax in range(2 * n) if ax not in moved]
    block = np.transpose(tensor, moved + remaining).reshape(dt, dt, rest, rest)

    rho_rest = np.trace(block, axis1=0, axis2=1)
    post = np.zeros_like(block)
    post[0, 0] = rho_rest

    rest_shape = tuple((reversed_dims * 2)[ax] for ax in remaining)
    post = post.reshape(local_dims + local_dims + rest_shape)
    inverse_perm = np.argsort(moved + remaining)
    return np.transpose(post, inverse_perm).reshape(rho.shape)


def _apply_unitary(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Apply a local matrix to every unitary column."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    columns = state.shape[1]
    tensor = state.reshape(reversed_dims + (columns,))
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    target_axes = [n - 1 - q for q in targets]
    return _contract_local(m, tensor, target_axes, n + 1, k).reshape(state.shape)


def _apply_superoperator(
    rho: np.ndarray,
    matrix: np.ndarray,
    targets: Sequence[int],
    dims: Sequence[int],
    reversed_dims: tuple[int, ...],
) -> np.ndarray:
    """Apply a local matrix sandwich to every superoperator column."""
    n = len(dims)
    k = len(targets)
    local_dims = tuple(dims[t] for t in targets)
    columns = rho.shape[1]
    tensor = rho.reshape(reversed_dims * 2 + (columns,))
    m = np.asarray(matrix, dtype=complex).reshape(local_dims + local_dims)
    total = 2 * n + 1
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    tensor = _contract_local(m, tensor, ket_axes, total, k)
    tensor = _contract_local(m.conj(), tensor, bra_axes, total, k)
    return tensor.reshape(rho.shape)
