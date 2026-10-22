"""Shared NumPy numerical helpers for matrix representations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod

import numpy as np

from ....._backends.steps import ApplyChannelStep, ResolvedStep
from .....implementation.matrices import shift_matrix
from .....noise.base import _sampled_unitary_branches
from ._measurement import _project_statevector, _statevector_probabilities

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


def _reset_density_matrix(
    rho: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Trace out targets and reprepare zero without consuming the supplied RNG.

    ``rho' = |0..0><0..0|_targets (x) Tr_targets(rho)`` is trace-preserving;
    no renormalization is needed. The generator is accepted for interface parity.
    """
    assert len(indices) >= 1, "reset_subsystems requires at least one index"
    reversed_dims = tuple(reversed(dims))
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


def _apply_kraus_jump(
    source: np.ndarray,
    kraus_ops: tuple[np.ndarray, ...],
    targets: Sequence[int],
    rng: np.random.Generator,
    apply_local: Callable[[np.ndarray, np.ndarray, Sequence[int]], np.ndarray],
) -> np.ndarray:
    """Sample and normalize one Kraus branch, consuming one RNG draw.

    Each branch starts from a copy so an in-place local application cannot
    corrupt the source used by the remaining branches.
    """
    branches = [apply_local(source.copy(), kraus, targets) for kraus in kraus_ops]
    norms = np.array([np.real(np.vdot(b, b)) for b in branches])
    # CPTP guarantees the norms sum to <psi|psi> = 1; renormalize anyway so
    # rng.choice never rejects the distribution over float round-off.
    chosen = int(rng.choice(len(branches), p=norms / norms.sum()))
    return branches[chosen] / np.sqrt(norms[chosen])


def _apply_kraus_sum(
    rho: np.ndarray,
    kraus_ops: tuple[np.ndarray, ...],
    targets: Sequence[int],
    apply_local: Callable[[np.ndarray, np.ndarray, Sequence[int]], np.ndarray],
) -> np.ndarray:
    """Sum local Kraus sandwiches for a density matrix or superoperator."""
    new = np.zeros_like(rho)
    for kraus in kraus_ops:
        new += apply_local(rho, kraus, targets)
    return new


def _sample_unitary(
    branches: tuple[np.ndarray, tuple[np.ndarray, ...], tuple[bool, ...]],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Draw one scaled-unitary branch, returning None for an identity.

    Probabilities are state-independent and consume exactly one RNG draw.
    Returned operators are already unit-norm; identity branches need no
    state access, application, or defensive copy.
    """
    probabilities, unitaries, identities = branches
    chosen = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
    if identities[chosen]:
        return None
    return unitaries[chosen]


def _reset_statevector(
    state: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one grouped branch, project, then shift its targets to zero.

    Consumes one RNG draw. Unmeasured subsystems retain the state conditioned
    on that outcome, including any remaining superposition.
    """
    assert len(indices) >= 1, "reset_subsystems requires at least one index"
    outcome_index = int(rng.choice(len(state), p=_statevector_probabilities(state)))
    state = _project_statevector(state, outcome_index, indices, dims)
    reversed_dims = tuple(reversed(dims))
    for index in indices:
        outcome = _digit(outcome_index, index, dims)
        if outcome != 0:
            inv = shift_matrix(dims[index], -outcome)
            state = _apply_statevector(state, inv, (index,), dims, reversed_dims)
    return state


def _reset_superoperator(
    state: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator | None = None,
    *,
    reset_channels: _ResetChannelCache,
) -> np.ndarray:
    """Apply cached reset channels to every operator column, without RNG draws.

    Each target uses ``sum_k |0><k| . |k><0|``, the operator representation of
    partial trace followed by preparation of zero. Cached Kraus matrices are
    reused across executions; the evolving operator is supplied explicitly.
    """
    assert len(indices) >= 1, "reset_subsystems requires at least one index"
    reversed_dims = tuple(reversed(dims))
    for index in indices:
        channel = reset_channels.resolve(index, dims[index])
        result = np.zeros_like(state)
        for kraus in channel.kraus_ops:
            result += _apply_superoperator(
                state, kraus, channel.target_indices, dims, reversed_dims
            )
        state = result
    return state
