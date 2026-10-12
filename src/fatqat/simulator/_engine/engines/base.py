"""Complete matrix-engine contract and shared Python execution coordination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any, Generic, Literal

import numpy as np

from ..._execution_contract import (
    RawResult,
    _InitialClassicalState as InitialClassicalState,
    _InitialEvolutionState as InitialEvolutionState,
    _SimulationConfig as SimulationConfig,
    _EngineCapabilities,
    _KernelCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
    _ExecutionContext as ExecutionContext,
)
from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    LossStep,
    MeasurementStep,
    PutStep,
    ResetStep,
    ResolvedStep,
)
from ....result import reduce_to_counts
from .._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
    _materialization_policy,
    _resolve_execution_policy,
)
from ..parallel import _run_shots_in_processes
from ..state import EvolutionState, QuantumDataT, QuantumState
from ..workflows.python import (
    _PreparedPythonExecution,
    _run_operator_plan,
    _run_single_pass,
    _run_per_shot,
    _run_shot_batch,
)


class MatrixEngine(ABC, Generic[QuantumDataT]):
    """Runtime-native numerical operations with shared execution coordination.

    The default preparation and local entries serve Python workflows. Engines
    can override them for compiled execution without changing the run interface.
    """

    # Concrete engines declare all support; numeric/execution helpers may be shared.
    _quantum_capabilities: _QuantumCapabilities
    _trajectory_capabilities: _TrajectoryCapabilities
    _kernel_capabilities: _KernelCapabilities
    _supports_process_shots: bool = False

    def __init__(self, name: str):
        self.name = name

        self._evolution_state: EvolutionState[QuantumDataT] | None = None
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def _state(self) -> QuantumDataT | None:
        """Forward existing kernel access to the owned quantum data."""
        if self._evolution_state is None:
            return None
        return self._evolution_state.quantum.data

    @_state.setter
    def _state(self, value: QuantumDataT | None) -> None:
        """Replace the quantum data without losing current classical state."""
        if value is None:
            self._evolution_state = None
        elif self._evolution_state is None:
            self._evolution_state = EvolutionState(
                QuantumState(value, self._state_field)
            )
        else:
            self._evolution_state.quantum.data = value

    @property
    def state(self) -> QuantumDataT:
        """Expose internal runtime storage without requesting a host export."""
        if self._state is None:
            raise RuntimeError("MatrixEngine state has not been initialized.")
        return self._state

    @state.setter
    def state(self, value: QuantumDataT) -> None:
        self._state = value

    @property
    def n_subsystems(self) -> int:
        return len(self._dims)

    @property
    def _state_field(self) -> str:
        """Use the declared representation for evolving state and results."""
        return self._quantum_capabilities.representation

    @property
    def capabilities(self) -> _EngineCapabilities:
        """Return supported quantum, classical, and numerical execution."""
        return _EngineCapabilities(
            quantum=self._quantum_capabilities,
            trajectory=self._trajectory_capabilities,
            kernels=self._kernel_capabilities,
            supports_process_shots=self._supports_process_shots,
        )

    @abstractmethod
    def _select_workflow(
        self,
        plan: Sequence[ResolvedStep],
        simulation: SimulationConfig,
        *,
        execution_path: str,
        counts_requested: bool,
        state_requested: bool,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> tuple[str, bool]:
        """Return the engine's workflow identifier and shot-threading support.

        Select against the complete request and controls before scheduling.
        The base passes the identifier through; concrete entries interpret it.
        """

    def resolve_execution_policy(
        self,
        plan: Sequence[ResolvedStep],
        simulation: SimulationConfig,
        *,
        counts_requested: bool,
        state_requested: bool,
        shots: int,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> ExecutionPolicy:
        """Choose execution paths for a plan using this engine's support."""
        execution_path = self._select_execution_path(plan, initial_state=initial_state)
        workflow, supports_shot_threads = self._select_workflow(
            plan,
            simulation,
            execution_path=execution_path,
            counts_requested=counts_requested,
            state_requested=state_requested,
            initial_state=initial_state,
        )
        capabilities = self.capabilities
        return _resolve_execution_policy(
            simulation,
            execution_path=execution_path,
            counts_requested=counts_requested,
            state_requested=state_requested,
            capabilities=capabilities.kernels,
            workflow=workflow,
            supports_shot_threads=supports_shot_threads,
            supports_process_shots=capabilities.supports_process_shots,
            shots=shots,
            plan_is_empty=not plan,
        )

    def _select_execution_path(
        self,
        plan: Sequence[ResolvedStep],
        *,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> Literal["operator", "single_pass", "per_shot"]:
        """Classify the required evolution shape from plan and engine semantics."""
        quantum = self.capabilities.quantum
        if quantum.is_operator:
            return "operator"
        # Only the trajectory executor handles occupancy and erasure reports.
        if initial_state.classical.occupied is not None:
            return "per_shot"
        measured: set[int] = set()
        for step in plan:
            if getattr(step, "condition", None) is not None or isinstance(
                step, (PutStep, LossStep)
            ):
                return "per_shot"
            if isinstance(step, MeasurementStep):
                if measured.intersection(step.measured_indices):
                    return "per_shot"
                measured.update(step.measured_indices)
                continue
            if isinstance(step, ResetStep):
                targets = step.reset_indices
            else:
                targets = step.target_indices
            if (
                isinstance(step, (ResetStep, ApplyChannelStep))
                and quantum.nonunitary_is_stochastic
            ):
                return "per_shot"
            if measured.intersection(targets):
                return "per_shot"
        return "single_pass"

    def configure_system(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions without allocating an evolving state."""
        self._set_dims(system_dims)
        self._n_clbits = int(n_clbits)
        self._state = None

    @abstractmethod
    def initialize(
        self,
        system_dims: Sequence[int],
        n_clbits: int = 0,
        *,
        initial_state: InitialEvolutionState[QuantumDataT] | None = None,
    ) -> None:
        """Initialize owned quantum and classical storage from validated input."""

    def _set_dims(self, system_dims: Sequence[int]) -> None:
        """Set ``_dims`` and its cached reverse together, so they never drift apart."""
        self._dims = tuple(int(d) for d in system_dims)
        self._reversed_dims = tuple(reversed(self._dims))

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        """Retain only the effective plan's identity-keyed step resolutions.

        Materialization calls this before resolving or compiling the plan.
        Shot initialization leaves these entries intact for trajectory reuse.
        Engines without step caches have nothing to prune.
        """

    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> Any:
        """Retain the effective plan and finalized deferred measurements.

        The payload is opaque at the dispatch boundary: overrides may return
        their own records paired with their local and shot-batch entries. This
        default returns ``_PreparedPythonExecution`` for the Python workflows.

        Overrides establish readiness for runtime-specific preparation before
        execution consumes it, retaining resources throughout pending work.
        The later execution scope does not retroactively cover preparation.
        """
        self.configure_system(system_dims, n_clbits)
        self._retain_step_caches(plan)
        measurements = (
            tuple(
                pair
                for step in plan
                if isinstance(step, MeasurementStep)
                for pair in zip(step.measured_indices, step.classical_indices)
            )
            if policy.execution_path == "single_pass"
            else ()
        )
        return _PreparedPythonExecution(plan, measurements)

    def _execution_scope(self, policy: ExecutionPolicy) -> AbstractContextManager[None]:
        """Set up the runtime context for local or shot-batch execution.

        Overrides keep dependent work ordered and its resources alive. Successful
        exit completes this invocation's numerical work, transfers, required
        classical updates, and preparation dependencies, even without host output.
        Unrelated device work need not finish.

        On failure, clean up and propagate execution errors. If cleanup also
        fails, keep the original error inspectable. Quantum state may outlive
        the scope. This does not require per-gate synchronization or make
        concurrent calls safe.
        """
        return nullcontext()

    def execute(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        context: ExecutionContext,
        policy: ExecutionPolicy,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Materialize and dispatch a plan through local or CPU process execution."""
        payload = self.materialize_execution(
            plan,
            system_dims=context.system_dims,
            n_clbits=context.n_clbits,
            policy=_materialization_policy(policy),
        )
        if policy.shot_strategy in ("none", "serial", "threads"):
            return self.execute_local(
                context, payload, policy, initial_state=initial_state
            )

        if policy.shot_strategy != "processes":
            raise RuntimeError(f"Unknown shot strategy: {policy.shot_strategy!r}")
        snapshots = _run_shots_in_processes(
            self._process_engine_factory(), context, payload, policy, initial_state
        )
        rows = np.asarray(snapshots, dtype=int).reshape(
            (len(snapshots), context.n_clbits)
        )
        outcome_keys, outcome_counts = reduce_to_counts(rows)
        return RawResult(
            outcome_keys=outcome_keys,
            outcome_counts=outcome_counts,
        )

    def _process_engine_factory(self) -> Callable[[], MatrixEngine[QuantumDataT]]:
        """Carry constructor configuration to workers without live execution state.

        Engines with additional constructor options override this hook.
        """
        return partial(type(self), name=self.name)

    def execute_local(
        self,
        context: ExecutionContext,
        payload: Any,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Execute a materialized state or operator plan through Python."""
        assert policy.shot_strategy != "processes"
        self.configure_system(context.system_dims, context.n_clbits)
        with self._execution_scope(policy):
            if policy.execution_path == "per_shot":
                return _run_per_shot(
                    self, payload.plan, context, initial_state=initial_state
                )
            if policy.execution_path == "operator":
                assert policy.shot_strategy == "none"
                self.initialize(
                    self._dims,
                    self._n_clbits,
                    initial_state=initial_state,
                )
                return _run_operator_plan(
                    self,
                    payload.plan,
                    np.random.default_rng(context.seed),
                    context.request,
                )
            assert policy.execution_path == "single_pass"
            return _run_single_pass(
                self,
                payload.plan,
                payload.deferred_measurements,
                context.shots,
                np.random.default_rng(context.seed),
                context.request,
                initial_state,
            )

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload: Any,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> list[tuple[int, ...]]:
        """Non-dispatching worker entry for one ordered batch of shots."""
        assert policy.shot_strategy == "serial"
        assert policy.kernel_strategy == "serial"
        self.configure_system(context.system_dims, context.n_clbits)
        with self._execution_scope(policy):
            return _run_shot_batch(
                self,
                payload.plan,
                seed_batch,
                initial_state,
            )

    @abstractmethod
    def measure_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> tuple[int, ...]: ...

    def measure_subsystem(self, index: int, rng: np.random.Generator) -> int:
        """
        Measure a single subsystem and return the result.
        """
        return self.measure_subsystems([index], rng)[0]

    @abstractmethod
    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None: ...

    def reset_subsystem(self, index: int, rng: np.random.Generator) -> None:
        """
        Reset a single subsystem to the |0> state.
        """
        self.reset_subsystems([index], rng)

    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """Return the computational-basis probability distribution of the state."""

    @abstractmethod
    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project the internal state, return the flat index."""

    @abstractmethod
    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply a single matrix step to the internal state in place."""

    @abstractmethod
    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Apply a Kraus channel to the internal state in place.

        Consumes ``rng`` only under statevector semantics (one draw to sample
        the trajectory branch); the density-matrix kernel is deterministic
        and accepts it for interface parity, like reset.
        """

    def _expectation_values(
        self,
        state: QuantumDataT,
        observables: Sequence[tuple[tuple[float, tuple[tuple[int, str], ...]], ...]],
        *,
        policy: ExecutionPolicy,
    ) -> tuple[float, ...]:
        """Evaluate Pauli sums on native state without modifying it.

        Factors use engine subsystem indices. Return host scalars and keep
        numerical work within the runtime's execution scope.
        """
        raise NotImplementedError

    def export_state(self) -> QuantumDataT:
        """Return state data in runtime-native storage for further processing.

        The default borrows internal data without copying or host transfer.
        Overrides convert the representation only when needed. Callers copy or
        transfer the result when they need independent storage or host data.
        """
        return self._export_state_data(self.state)

    def _export_state_data(self, data: QuantumDataT) -> QuantumDataT:
        """Convert representation when required, retaining runtime storage."""
        return data

    def _export_evolution_state(self) -> InitialEvolutionState[QuantumDataT]:
        """Export completed local evolution as reusable initialization.

        Quantum data is borrowed in runtime-native storage. Classical values
        are frozen so later execution cannot change the saved initialization.
        Each consumer initializes its own mutable state before evolving it.
        """
        if self._evolution_state is None:
            raise RuntimeError("MatrixEngine state has not been initialized.")
        classical = self._evolution_state.classical
        return InitialEvolutionState(
            quantum=self.export_state(),
            classical=InitialClassicalState(
                clbits=None if classical.clbits is None else tuple(classical.clbits),
                occupied=(
                    None
                    if classical.occupied is None
                    else frozenset(classical.occupied)
                ),
            ),
        )

    @abstractmethod
    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Return sampled flat basis-state indices as a NumPy array."""
