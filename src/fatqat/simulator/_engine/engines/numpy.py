"""NumPy engines for the matrix backend family.

`NumpySVEngine` (statevector) and `NumpyDMEngine` (density matrix) are the
two *state* `MatrixEngine` implementations. `MatrixEngine` owns shared plan
materialization and Python execution; `_NumpyMatrixEngine` supplies NumPy
initialization, sampling, expectations, and measurement decoding. Each leaf
contributes its numerical operations and explicit capabilities. Engine-owned
selection determines the semantic execution shape.

`NumpyUnitaryEngine` and `NumpySuperopEngine` are the two *operator* engines:
they compute the program's map rather than a state under it. Each is its state
twin evolved on many columns at once - a unitary is ``size`` statevector
columns (column ``j`` is ``U|j>``), a super-operator is ``size**2``
density-matrix columns (column ``b`` is the image of basis matrix ``E_b``).
`_NumpyOperatorEngine` rejects sampling; shared execution selects the operator pass.

Conventions:

- private little-endian engine order: engine subsystem ``q`` has place value
  ``prod(dims[:q])``. The simulator maps public subsystems into this order, so
  the native flat buffer is already public-order; exporting it with another
  subsystem reversal would be wrong and would add a full-result copy.
- an ``ApplyMatrixStep``'s ``target_indices`` map to the matrix's local index
  with ``target_indices[0]`` as the most-significant digit.
- a density matrix reshaped to ``reversed(dims) + reversed(dims)`` exposes ket
  axes ``[0, n)`` and bra axes ``[n, 2n)``; subsystem ``q``'s ket axis is
  ``n - 1 - q`` and its bra axis is ``2n - 1 - q``.

Semantics differences:

- statevector evolution is ``|psi'> = U_T |psi>``; density-matrix evolution is
  ``rho' = U_T rho U_T^dagger``. Neither materializes the global ``U_T``.
- measurement is trajectory-style on both: Born sampling, projective collapse,
  normalization.
- reset samples a branch on the statevector (one rng draw), so any reset forces
  the per-shot path; on the density matrix it is the deterministic
  partial-trace channel (no rng draw), so an unconditional reset stays on the
  fast path. Per-shot rng streams are therefore not draw-for-draw aligned across
  the two, so counts for a fixed seed are reproducible per engine, not across
  engines.
- a channel (``ApplyChannelStep``) follows the same split as reset: the
  statevector samples one Kraus branch per occurrence (quantum-jump
  unravelling, one rng draw), forcing the per-shot path; the density matrix
  applies the exact Kraus sum ``sum_i K_i rho K_i^dagger`` (no rng draw), so
  an unconditional channel stays on the fast path.
- a channel of scaled unitaries (every Pauli channel; also `Depolarizing` and
  `PhaseDamping` in any dimension) has state-independent branch probabilities,
  so the statevector draws its branch from that fixed vector and applies only
  the operator it drew. One rng draw either way, so which sampler a channel
  takes affects neither the stream nor path classification.
- measurement reporting is identical on both: the collapse keeps the physical
  outcome, maps it to a reported digit, then optionally resamples that digit
  through ``MeasurementStep.confusions``. It never affects path
  classification. On the per-shot path feedforward conditions read the
  (possibly confused) reported value.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from math import prod

import numpy as np

from ...._expectation import expectation_density_matrix, expectation_statevector
from ..._execution_contract import (
    _InitialEvolutionState as InitialEvolutionState,
    _SimulationConfig as SimulationConfig,
    _KernelCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
)
from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    ResolvedStep,
)
from .._execution_policy import _ExecutionPolicy as ExecutionPolicy
from .base import MatrixEngine
from ..numerics.numpy import (
    _ChannelRouteCache,
    _ResetChannelCache,
    _allocate_statevector,
    _allocate_density_matrix,
    _allocate_unitary,
    _allocate_superoperator,
    _apply_density_matrix,
    _apply_kraus_jump,
    _apply_kraus_sum,
    _apply_statevector,
    _apply_superoperator,
    _apply_unitary,
    _density_matrix_probabilities,
    _digit,
    _project_density_matrix,
    _reset_density_matrix,
    _reset_statevector,
    _project_statevector,
    _statevector_probabilities,
    _sample_unitary,
)
from ..state import ClassicalState, EvolutionState, QuantumState

# --- shared NumPy operations ---


class _NumpyMatrixEngine(MatrixEngine[np.ndarray]):
    """NumPy allocation, sampling, measurement decoding, and expectations.

    Shared plan preparation and workflow invocation come from ``MatrixEngine``.
    Subclasses supply representation-specific numerical operations.
    """

    def _select_workflow(
        self,
        plan: Sequence[ResolvedStep],
        simulation: SimulationConfig,
        *,
        execution_path: str,
        counts_requested: bool,
        state_requested: bool,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> tuple[str, bool]:
        """Use shared Python execution for states and operators."""
        return "python", False

    def initialize(
        self,
        system_dims: Sequence[int],
        n_clbits: int = 0,
        *,
        initial_state: InitialEvolutionState[np.ndarray] | None = None,
    ) -> None:
        self.configure_system(system_dims, n_clbits)
        initial = (
            initial_state if initial_state is not None else InitialEvolutionState()
        )
        quantum = self._allocate(prod(self._dims) if self._dims else 1, initial.quantum)
        clbits = initial.classical.clbits
        occupied = initial.classical.occupied
        self._evolution_state = EvolutionState(
            QuantumState(quantum, self._state_field),
            ClassicalState(
                clbits=None if clbits is None else list(clbits),
                occupied=None if occupied is None else set(occupied),
            ),
        )

    def _expectation_values(
        self,
        state: np.ndarray,
        observables: Sequence[tuple[tuple[float, tuple[tuple[int, str], ...]], ...]],
        *,
        policy: ExecutionPolicy,
    ) -> tuple[float, ...]:
        kernel = (
            expectation_statevector
            if self._state_field == "statevector"
            else expectation_density_matrix
        )
        values = []
        with self._execution_scope(policy):
            for terms in observables:
                kernel_terms = tuple(
                    (
                        coefficient,
                        tuple(
                            (self.n_subsystems - 1 - index, letter)
                            for index, letter in factors
                        ),
                    )
                    for coefficient, factors in terms
                )
                values.append(kernel(state, kernel_terms))
        return tuple(values)

    @abstractmethod
    def _allocate(self, size: int, initial_state: np.ndarray | None) -> np.ndarray:
        """Return a fresh owned state over ``size`` basis states."""

    def measure_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> tuple[int, ...]:
        assert len(indices) >= 1, "measure_subsystems requires at least one index"
        flat = self.collapse(indices, rng)
        return tuple(_digit(flat, index, self._dims) for index in indices)

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample flat basis-state indices from the NumPy quantum buffer."""
        return rng.choice(self.state.shape[0], size=shots, p=self.probabilities())


# --- statevector engine ---


class NumpySVEngine(_NumpyMatrixEngine):
    """State-vector engine: evolves ``|psi>`` as a flat little-endian array."""

    _quantum_capabilities = _QuantumCapabilities(
        "statevector", supports_nonunitary=True, nonunitary_is_stochastic=True
    )
    _trajectory_capabilities = _TrajectoryCapabilities(
        classical_register=True, occupancy=True
    )
    _supports_process_shots = True
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
    )

    def __init__(self, name: str = "numpy-sv"):
        super().__init__(name)
        self._channel_routes = _ChannelRouteCache()

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        self._channel_routes.retain(plan)

    _allocate = staticmethod(_allocate_statevector)

    def apply(self, step: ApplyMatrixStep) -> None:
        self._state = self._apply_local(self.state, step.matrix, step.target_indices)

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Sample one Kraus branch, by the cheapest sampler the operators allow.

        A channel of scaled unitaries draws from a fixed probability vector;
        everything else takes the general quantum-jump path. Both consume
        exactly one rng draw.
        """
        branches = self._channel_routes.resolve(step)
        if branches is None:
            self._apply_kraus_jump(step, rng)
        else:
            unitary = _sample_unitary(branches, rng)
            if unitary is not None:
                self._state = self._apply_local(
                    self.state, unitary, step.target_indices
                )

    def _apply_kraus_jump(
        self, step: ApplyChannelStep, rng: np.random.Generator
    ) -> None:
        """Quantum-jump unravelling for a channel of general Kraus operators.

        Each candidate branch ``K_i |psi>`` is produced by the same local-apply
        primitive gates use; its squared norm is the branch probability. One
        branch is drawn and kept, normalized.

        Each branch starts from a fresh copy of the state: reusing
        ``self.state`` across branches would let an in-place kernel (the Numba
        subclass's) corrupt the source mid-loop.
        """
        self._state = _apply_kraus_jump(
            self.state, step.kraus_ops, step.target_indices, rng, self._apply_local
        )

    def _apply_local(
        self, state: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Apply a local matrix to flat ``targets``, returning the new flat state."""
        return _apply_statevector(
            state, matrix, targets, self._dims, self._reversed_dims
        )

    def probabilities(self) -> np.ndarray:
        return _statevector_probabilities(self.state)

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        state = self.state
        idx = int(rng.choice(len(state), p=self.probabilities()))
        self._state = _project_statevector(state, idx, measured_subsystems, self._dims)
        return idx

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None:
        """Measure ``indices`` as one branch, then shift each nonzero digit to ``|0>``.

        Consumes exactly one rng draw; the rest of an entangled state is left
        correctly conditioned on the sampled branch.
        """
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        idx = self.collapse(indices, rng)
        self._state = _reset_statevector(
            self._state, idx, indices, self._dims, self._apply_local
        )


# --- density-matrix engine ---


class NumpyDMEngine(_NumpyMatrixEngine):
    """Density-matrix engine: evolves ``rho`` as a ``(size, size)`` matrix."""

    _quantum_capabilities = _QuantumCapabilities(
        "density_matrix", supports_nonunitary=True, nonunitary_is_stochastic=False
    )
    _trajectory_capabilities = _TrajectoryCapabilities(
        classical_register=True, occupancy=True
    )
    _supports_process_shots = True
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
    )

    def __init__(self, name: str = "numpy-dm"):
        super().__init__(name)

    _allocate = staticmethod(_allocate_density_matrix)

    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply ``rho' = U_T rho U_T^dagger`` without materializing ``U_T``."""
        self._state = self._apply_local_sandwich(
            self.state, step.matrix, step.target_indices
        )

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Apply the exact Kraus sum ``rho' = sum_i K_i rho K_i^dagger``.

        Each term reuses the two-sided local-apply primitive gates use; the
        branch weights are implicit in the ``K_i rho K_i^dagger`` sandwiches,
        which is what makes this exact. No randomness is consumed (``rng`` is
        accepted only for interface parity, like reset).
        """
        self._state = _apply_kraus_sum(
            self.state, step.kraus_ops, step.target_indices, self._apply_local_sandwich
        )

    def _apply_local_sandwich(
        self, rho: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Return ``M_T rho M_T^dagger`` for a local matrix on flat ``targets``.

        ``rho`` is viewed as a 2n-axis ket/bra tensor and ``M`` (resp.
        ``conj(M)``) is contracted into the target ket (resp. bra) axes. The
        matrix need not be unitary: gates and Kraus operators share this
        primitive, differing only in whether the caller sums over terms.
        """
        return _apply_density_matrix(
            rho, matrix, targets, self._dims, self._reversed_dims
        )

    def probabilities(self) -> np.ndarray:
        return _density_matrix_probabilities(self.state)

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        rho = self.state
        size = rho.shape[0]
        idx = int(rng.choice(size, p=self.probabilities()))
        self._state = _project_density_matrix(rho, idx, measured_subsystems, self._dims)
        return idx

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator | None = None
    ) -> None:
        """Deterministically reset ``indices``: partial trace, then reprepare ``|0>``.

        ``rho' = |0..0><0..0|_targets (x) Tr_targets(rho)``. Trace-preserving, so
        no renormalization is needed and no randomness is consumed (``rng`` is
        accepted only for interface parity).
        """
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        self._state = _reset_density_matrix(
            self.state, indices, self._dims, self._reversed_dims
        )


# --- operator engines ---


class _NumpyOperatorEngine(_NumpyMatrixEngine):
    """Sampling rejection for NumPy operator representations.

    ``MatrixEngine`` runs their deterministic Python operator workflow;
    the concrete engines supply operator allocation and numerical operations.
    """

    def probabilities(self) -> np.ndarray:
        """Unsupported: an operator is a map, not a distribution over states."""
        raise NotImplementedError(
            f"{self._state_field} execution has no basis-state distribution"
        )

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Unsupported: an operator cannot be projected onto one outcome."""
        raise NotImplementedError(
            f"{self._state_field} execution cannot represent a measurement"
        )


# Pylint treats the inherited probabilities/collapse methods, which deliberately
# raise NotImplementedError, as abstract. Operator engines reuse state kernels
# but intentionally do not support sampling or measurement.
class NumpyUnitaryEngine(  # pylint: disable=abstract-method
    _NumpyOperatorEngine, NumpySVEngine
):
    """Unitary engine: evolves ``U`` as a ``(size, size)`` operator matrix.

    Column ``j`` of ``U`` is the statevector ``U|j>``, so the whole operator is
    the statevector kernel run on ``size`` columns at once.
    """

    _quantum_capabilities = _QuantumCapabilities(
        # Retain the SV branch classification; non-unitary plans are rejected.
        "unitary",
        supports_nonunitary=False,
        nonunitary_is_stochastic=True,
    )
    _trajectory_capabilities = _TrajectoryCapabilities()
    _supports_process_shots = False
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
    )

    def __init__(self, name: str = "numpy-unitary"):
        super().__init__(name)

    _allocate = staticmethod(_allocate_unitary)

    def _apply_local(
        self, state: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Apply a local matrix to every column of the operator at once."""
        return _apply_unitary(state, matrix, targets, self._dims, self._reversed_dims)


# As for NumpyUnitaryEngine, the inherited sampling methods are unsupported,
# rather than missing implementations in this concrete operator engine.
class NumpySuperopEngine(  # pylint: disable=abstract-method
    _NumpyOperatorEngine, NumpyDMEngine
):
    """Super-operator engine: evolves ``S`` as a ``(size**2, size**2)`` matrix.

    Internally, column ``b`` of ``S`` is the row-stacked image of basis matrix
    ``E_b``, so the whole channel is the density-matrix kernel run on
    ``size**2`` columns at once (``S = kron(M, conj(M))``). :meth:`export_state`
    converts the completed matrix once to the public column-stacking
    convention.
    """

    _quantum_capabilities = _QuantumCapabilities(
        "superop", supports_nonunitary=True, nonunitary_is_stochastic=False
    )
    _trajectory_capabilities = _TrajectoryCapabilities()
    _supports_process_shots = False
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
    )

    def __init__(self, name: str = "numpy-superop"):
        super().__init__(name)
        self._reset_channels = _ResetChannelCache()

    _allocate = staticmethod(_allocate_superoperator)

    def _export_state_data(self, data: np.ndarray) -> np.ndarray:
        """Convert NumPy super-operator data to public column stacking.

        This existing vectorization conversion does not reorder subsystems.
        """
        size = prod(self._dims) if self._dims else 1
        internal = data.reshape((size,) * 4)
        exported = np.empty_like(data, order="C")
        exported.reshape((size,) * 4)[...] = internal.transpose(1, 0, 3, 2)
        return exported

    def _apply_local_sandwich(
        self, rho: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Return ``M_T . M_T^dagger`` applied to every column at once."""
        return _apply_superoperator(
            rho, matrix, targets, self._dims, self._reversed_dims
        )

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator | None = None
    ) -> None:
        """Reset ``indices`` through the Kraus channel ``sum_k |0><k| . |k><0|``.

        Equal to the density matrix's partial-trace reset. Deterministic -
        ``rng`` is accepted only for interface parity.
        """
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        for index in indices:
            self.apply_channel(
                self._reset_channels.resolve(index, self._dims[index]), rng
            )
