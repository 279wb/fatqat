"""Shared coset arithmetic for Numba matrix engines."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Gate-matrix structure codes. A diagonal or (phased) permutation matrix has at
# most one nonzero per row, so its application is a single multiply per amplitude
# instead of the D-term dot product a dense row needs - a machine-independent
# constant-factor win that covers most common gates (Z/S/T/RZ/CPhase/CZ are
# diagonal; X/CX/CCX/SWAP/iSwap are permutations).
_DENSE = 0
_DIAGONAL = 1
_PERMUTATION = 2
# CSR is selected during operand preparation, not by `_classify_matrix`.
_SPARSE = 3


@njit(cache=True)
def _classify_matrix(
    matrix: np.ndarray, columns: np.ndarray, values: np.ndarray
) -> int:  # pragma: no cover - compiled by Numba
    """Classify ``matrix`` and fill the single-nonzero-per-row description.

    For each row ``r`` records the column ``columns[r]`` of its (sole) nonzero and
    the value ``values[r]`` there. Returns ``_DIAGONAL`` if every nonzero is on
    the diagonal, ``_PERMUTATION`` if each row has exactly one nonzero (off the
    diagonal somewhere), else ``_DENSE`` (``columns`` / ``values`` unused). Gate
    matrices are analytic, so structural zeros are exact - no tolerance needed.
    """
    d = matrix.shape[0]
    is_diagonal = True
    for r in range(d):
        count = 0
        column = 0
        for c in range(d):
            if matrix[r, c].real != 0.0 or matrix[r, c].imag != 0.0:
                count += 1
                column = c
        if count != 1:
            return _DENSE
        columns[r] = column
        values[r] = matrix[r, column]
        if column != r:
            is_diagonal = False
    return _DIAGONAL if is_diagonal else _PERMUTATION


@njit(cache=True)
def _spread_base(
    start: int, comp_strides: np.ndarray, comp_dims: np.ndarray, counter: np.ndarray
) -> int:  # pragma: no cover - compiled by Numba
    """Spread coset index ``start`` into odometer ``counter``; return its flat base."""
    base = 0
    remainder = start
    for i in range(comp_strides.shape[0]):
        digit = remainder % comp_dims[i]
        remainder = remainder // comp_dims[i]
        counter[i] = digit
        base += digit * comp_strides[i]
    return base


@njit(cache=True, inline="always")
def _advance_base(
    base: int, comp_strides: np.ndarray, comp_dims: np.ndarray, counter: np.ndarray
) -> int:  # pragma: no cover - compiled by Numba
    """Advance the odometer one coset, returning the next flat base."""
    for i in range(comp_strides.shape[0]):
        counter[i] += 1
        base += comp_strides[i]
        if counter[i] < comp_dims[i]:
            return base
        counter[i] = 0
        base -= comp_strides[i] * comp_dims[i]
    return base


@njit(cache=True)
def _apply_dense_range(
    state, matrix, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Dense ``D x D`` application to cosets ``[start, end)``: ``D``-term dot per row.

    Cosets partition the amplitudes by non-target ("complement") digits; within a
    coset the ``D`` target amplitudes sit at ``base + offsets[c]`` and the matrix
    maps them among themselves. ``base`` is spread from ``start`` once, then the
    range is walked with a division-free odometer; disjoint cosets keep the result
    independent of how the range is split.
    """
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    updated = np.empty(local_dim, dtype=np.complex128)
    for _ in range(start, end):
        for r in range(local_dim):
            acc = 0.0 + 0.0j
            for c in range(local_dim):
                acc += matrix[r, c] * state[base + offsets[c]]
            updated[r] = acc
        for r in range(local_dim):
            state[base + offsets[r]] = updated[r]
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True)
def _apply_diagonal_range(
    state, diagonal, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Diagonal application: scale each amplitude in place (no gather, no dot)."""
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    for _ in range(start, end):
        for r in range(local_dim):
            state[base + offsets[r]] *= diagonal[r]
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True)
def _apply_permutation_range(
    state, columns, values, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Permutation application: one gather then one scaled scatter per coset."""
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    gathered = np.empty(local_dim, dtype=np.complex128)
    for _ in range(start, end):
        for c in range(local_dim):
            gathered[c] = state[base + offsets[c]]
        for r in range(local_dim):
            state[base + offsets[r]] = values[r] * gathered[columns[r]]
        base = _advance_base(base, comp_strides, comp_dims, counter)


@njit(cache=True, inline="always")
def _apply_resolved_range(
    state,
    code,
    matrix,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    start,
    end,
    indptr=None,
) -> None:  # pragma: no cover - compiled by Numba
    """Route one coset range to the structure-specialized kernel.

    Dense uses the full ``matrix``; diagonal uses ``values[row]`` only.
    Permutation uses one source ``columns[row]`` and multiplier ``values[row]``
    per output row. CSR uses ``columns`` and ``values`` for its nonzero entries,
    with ``indptr`` delimiting rows. Non-CSR representations omit ``indptr``;
    each kernel ignores operands it does not use.
    """
    if code == _SPARSE and indptr is not None:
        _apply_sparse_range(
            state, indptr, columns, values, offsets, comp_strides, comp_dims, start, end
        )
    elif code == _DIAGONAL:
        _apply_diagonal_range(
            state, values, offsets, comp_strides, comp_dims, start, end
        )
    elif code == _PERMUTATION:
        _apply_permutation_range(
            state, columns, values, offsets, comp_strides, comp_dims, start, end
        )
    else:
        _apply_dense_range(state, matrix, offsets, comp_strides, comp_dims, start, end)


@njit(cache=True)
def _apply_resolved_serial(
    state,
    code,
    matrix,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    num_cosets,
    indptr=None,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Single-threaded application of an already-classified gate."""
    _apply_resolved_range(
        state,
        code,
        matrix,
        columns,
        values,
        offsets,
        comp_strides,
        comp_dims,
        0,
        num_cosets,
        indptr,
    )
    return state


@njit(cache=True, parallel=True)
def _apply_resolved_parallel(
    state,
    code,
    matrix,
    columns,
    values,
    offsets,
    comp_strides,
    comp_dims,
    num_cosets,
    n_chunks,
    indptr=None,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Parallel application of an already-classified gate over coset chunks."""
    for chunk in prange(n_chunks):  # pylint: disable=not-an-iterable
        start = chunk * num_cosets // n_chunks
        end = (chunk + 1) * num_cosets // n_chunks
        if start < end:
            _apply_resolved_range(
                state,
                code,
                matrix,
                columns,
                values,
                offsets,
                comp_strides,
                comp_dims,
                start,
                end,
                indptr,
            )
    return state


# --- sparse coset kernel (density-matrix channel super-operators) ---
#
# A channel super-operator ``sum_i kron(K_i, conj(K_i))`` is often mostly zero
# even when neither diagonal nor a permutation (two-qubit depolarizing fills
# ``2 d**2 - d`` of ``d**4`` entries), which `_classify_matrix` calls dense.
# These kernels walk a CSR of the exact nonzeros in column order - bit-identical
# to the dense walk, at a fraction of the multiplies.


@njit(cache=True)
def _apply_sparse_range(
    state, indptr, indices, data, offsets, comp_strides, comp_dims, start, end
) -> None:  # pragma: no cover - compiled by Numba
    """Sparse ``D x D`` (CSR) application to cosets ``[start, end)``."""
    local_dim = offsets.shape[0]
    counter = np.empty(comp_strides.shape[0], dtype=np.int64)
    base = _spread_base(start, comp_strides, comp_dims, counter)
    gathered = np.empty(local_dim, dtype=np.complex128)
    updated = np.empty(local_dim, dtype=np.complex128)
    for _ in range(start, end):
        for c in range(local_dim):
            gathered[c] = state[base + offsets[c]]
        for r in range(local_dim):
            acc = 0.0 + 0.0j
            for k in range(indptr[r], indptr[r + 1]):
                acc += data[k] * gathered[indices[k]]
            updated[r] = acc
        for r in range(local_dim):
            state[base + offsets[r]] = updated[r]
        base = _advance_base(base, comp_strides, comp_dims, counter)


def _apply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    plan: tuple,
    code: int,
    columns: np.ndarray,
    values: np.ndarray,
    *,
    sparse: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Apply a classified matrix to native storage using a precomputed layout.

    The layout describes flat storage, including column-batched operators and
    vectorized density matrices. Buffer conversion and flattening belong here;
    callers supply numerical operands without managing the kernel's flat view.
    Contiguous complex128 input is updated in place, with its shape preserved.
    CSR operands select sparse execution; otherwise ``code`` selects dense,
    diagonal, or permutation arithmetic. The layout controls kernel threading.
    """
    state = np.ascontiguousarray(state, dtype=np.complex128)
    flat = state if state.ndim == 1 else state.reshape(-1)
    offsets, comp_strides, comp_dims, num_cosets, n_chunks = plan
    indptr = None
    if sparse is not None:
        code = _SPARSE
        indptr, columns, values = sparse
    else:
        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
    if n_chunks <= 1:
        result = _apply_resolved_serial(
            flat,
            code,
            matrix,
            columns,
            values,
            offsets,
            comp_strides,
            comp_dims,
            num_cosets,
            indptr,
        )
    else:
        result = _apply_resolved_parallel(
            flat,
            code,
            matrix,
            columns,
            values,
            offsets,
            comp_strides,
            comp_dims,
            num_cosets,
            n_chunks,
            indptr,
        )
    return result if state.ndim == 1 else result.reshape(state.shape)
