"""NumPy state and operator buffer allocation."""

from __future__ import annotations

import numpy as np


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
