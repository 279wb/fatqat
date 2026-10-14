"""Complete matrix-engine contract and shared Python execution coordination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any, Generic

import numpy as np

from ..._execution_contract import (
    RawResult,
    _InitialClassicalState as InitialClassicalState,
    _InitialEvolutionState as InitialEvolutionState,
    _SimulationConfig as SimulationConfig,
    _EngineCapabilities,
    _ExecutionContext as ExecutionContext,
)
from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    MeasurementStep,
    ResolvedStep,
)
from ....result import reduce_to_counts
from .._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
    _materialization_policy,
)
from ..parallel import _run_shots_in_processes
from ..state import EvolutionState, QuantumDataT
from ..workflows.python import (
    _PreparedPythonExecution,
    _run_operator_plan,
    _run_single_pass,
    _run_one_shot,
)


class MatrixEngine(ABC, Generic[QuantumDataT]):
    """Runtime-native numerical operations with shared execution coordination.

    The default preparation and local entries serve Python workflows. Engines
    can override them for compiled execution without changing the run interface.
    """

    # Concrete engines declare all support; numeric/execution helpers may be shared.
    _capabilities: _EngineCapabilities

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
        return self._evolution_state.quantum

    @_state.setter
    def _state(self, value: QuantumDataT | None) -> None:
        """Replace the quantum data without losing current classical state."""
        if value is None:
            self._evolution_state = None
        elif self._evolution_state is None:
            self._evolution_state = EvolutionState(value)
        else:
            self._evolution_state.quantum = value

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
        return self.capabilities.quantum.representation

    @property
    def capabilities(self) -> _EngineCapabilities:
        """Return supported quantum, classical, and numerical execution."""
        return self._capabilities

    @abstractmethod
    def _plan_execution(
        self,
        plan: Sequence[ResolvedStep],
        *,
        context: ExecutionContext,
        simulation_config: SimulationConfig,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> ExecutionPolicy:
        """Validate the request and choose its workflow and scheduling.

        Implementations use shared request classification and validation, then
        resolve runtime-specific choices before preparing numerical payloads.
        """

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
        simulation_config: SimulationConfig,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Plan, materialize, and execute one requested evolution."""
        policy = self._plan_execution(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
        )
        return self._execute_plan(plan, context, policy, initial_state)

    def execute_expectation(
        self,
        plan: tuple[ResolvedStep, ...],
        observables: Sequence[tuple[tuple[float, tuple[tuple[int, str], ...]], ...]],
        *,
        context: ExecutionContext,
        simulation_config: SimulationConfig,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> tuple[float, ...]:
        """Evolve and contract Pauli sums within the selected runtime policy.

        Empty observables validate execution controls without evolving a state.
        """
        policy = self._plan_execution(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
        )
        if not observables:
            return ()
        raw = self._execute_plan(plan, context, policy, initial_state)
        if raw.state is None:
            raise RuntimeError("expectation execution returned no state")
        return self._expectation_values(raw.state, observables, policy=policy)

    def _execute_plan(
        self,
        plan: tuple[ResolvedStep, ...],
        context: ExecutionContext,
        policy: ExecutionPolicy,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Materialize and dispatch an engine-selected execution policy."""
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
            state_requested = context.request.final_state
            rows = None
            if policy.execution_path == "per_shot":
                rows = self._run_per_shot(
                    payload.plan,
                    context,
                    initial_state,
                    state_requested=state_requested,
                )
            else:
                rng = np.random.default_rng(context.seed)
                self.initialize(
                    context.system_dims,
                    context.n_clbits,
                    initial_state=initial_state,
                )
                if policy.execution_path == "operator":
                    assert policy.shot_strategy == "none"
                    _run_operator_plan(self, payload.plan, rng)
                else:
                    assert policy.execution_path == "single_pass"
                    assert self._evolution_state is not None
                    rows = _run_single_pass(
                        self,
                        payload.plan,
                        payload.deferred_measurements,
                        self._evolution_state.classical,
                        context,
                        rng,
                        state_requested=state_requested,
                    )
            outcome_keys, outcome_counts = (
                reduce_to_counts(rows) if rows is not None else (None, None)
            )
            state = self.export_state() if state_requested else None
            return RawResult(
                outcome_keys=outcome_keys, outcome_counts=outcome_counts, state=state
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
            return self._run_shot_batch(
                payload.plan, context, seed_batch, initial_state
            )

    def _run_per_shot(
        self,
        plan: Sequence[ResolvedStep],
        context: ExecutionContext,
        initial_state: InitialEvolutionState[QuantumDataT],
        *,
        state_requested: bool,
    ) -> np.ndarray | None:
        """Assign local trajectory streams and return requested classical samples."""
        n_iters = context.shots if context.request.counts else int(state_requested)
        snapshots = self._run_shot_batch(
            plan,
            context,
            np.random.SeedSequence(context.seed).spawn(n_iters),
            initial_state,
        )
        if not context.request.counts:
            return None
        return np.asarray(snapshots, dtype=int).reshape(
            (len(snapshots), context.n_clbits)
        )

    def _run_shot_batch(
        self,
        plan: Sequence[ResolvedStep],
        context: ExecutionContext,
        seed_sequences: Sequence[np.random.SeedSequence],
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> list[tuple[int, ...]]:
        """Initialize each trajectory within the caller's active runtime scope.

        Local and process entries share this lifecycle. The workflow borrows the
        current classical container; the engine retains quantum storage and caches.
        """
        snapshots = []
        for seed_sequence in seed_sequences:
            self.initialize(
                context.system_dims,
                context.n_clbits,
                initial_state=initial_state,
            )
            assert self._evolution_state is not None
            snapshots.append(
                _run_one_shot(
                    self,
                    plan,
                    self._evolution_state.classical,
                    context,
                    np.random.default_rng(seed_sequence),
                )
            )
        return snapshots

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
        return self.state

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
