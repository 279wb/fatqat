"""Numba matrix-family engines.

`NumbaSVEngine` (statevector) and `NumbaDMEngine` (density matrix) reuse
shared materialization and Python execution from ``MatrixEngine`` and
``initialize`` / ``measure_subsystems`` operations from their NumPy twins,
while replacing numeric kernels with Numba-jitted loops. `NumbaUnitaryEngine`
and `NumbaSuperopEngine` then reuse *those* kernels for the operator
representations, overriding only the apply plan (see the operator-engine
section at the bottom). All four are reachable via
``Simulator(method=..., runtime="numba")``.

Kernel selection is key-driven: an `ApplyMatrixStep` carries the canonical
identity of the implementation that produced its matrix (``kernel_key``), and
each engine resolves which specialized kernel a step gets once per plan - by
declared identity or one `_classify_matrix` content scan, never per application
or per shot. `_KERNEL_SPECS` is the statevector key-to-structure table; the
density matrix derives its table from the same declarations because the
Kronecker product preserves structure (see below).

- `NumbaSVEngine` replaces gate application (`apply` / `_apply_local`),
  `probabilities`, `sample_indices`, `collapse`, and `reset_subsystems`.
  ``measure_subsystems`` delegates to collapse; reset uses direct digit shifts
  shared with the compiled trajectory path.
- `NumbaDMEngine` replaces `apply` / `apply_channel`, `probabilities`,
  `sample_indices`, and `collapse`. It reuses the *same* coset-walk kernels as
  the statevector path by viewing ``rho`` (shape ``(size, size)``) as a flat
  vector over a doubled ``2n``-subsystem system: ``n`` bra subsystems (strides
  ``prod(dims[:q])``) then ``n`` ket subsystems (strides ``size *
  prod(dims[:q])``). The sandwich ``M rho M^dagger`` is the single
  super-operator ``kron(M, conj(M))`` on ``vec(rho)``, and a channel is
  ``sum_i kron(K_i, conj(K_i))`` - each one coset walk over the combined
  ket+bra super-target. The Kronecker product preserves structure, so a gate's
  declared identity transfers to its super-operator: a declared-dense key skips
  the scan, everything else takes one scan of the super-operator, cached per
  step. ``4^n`` is memory-bound and each gate is one pass, so DM parallelizes
  later than the statevector path (`_MIN_SIZE_TO_THREAD_DM`). Reset stays the
  inherited NumPy partial-trace channel.

Noise math lives in ``noise.nb`` (quantum-jump branch selection for the
statevector, the channel super-operator for the density matrix, readout-confusion
resampling, and the compiled multi-shot plan flattening); *applying* a Kraus operator
is a plain local-matrix application on the coset kernels here. This module
imports ``noise.nb``; the reverse is forced never to happen (see its docstring).

The compiled multi-shot statevector channel path weighs branches the way Aer's trajectory
sampler does, not the way the NumPy reference does: instead of materializing
every ``K_i|psi>`` and norming it, it forms the targets' reduced density matrix
``rho_T`` once (`_reduced_density`) and reads each branch probability off it as
``Tr(K_i^dagger K_i rho_T)``, then applies only the chosen operator (see
`_apply_channel_step`). A mathematically equal but numerically distinct estimator, so
it stays seed-reproducible and agrees with NumPy in distribution, not per-seed
bit-for-bit.

The operator engines do not apply gates one at a time. An operator buffer is
``(rows, columns)`` and every step acts on the row index alone, so columns are
independent and a whole plan runs inside one parallel region split over column
blocks (``workflows.numba.operators``). Those loops keep the per-step
accumulation order, so they are bit-identical to the per-gate coset kernels.
When ``fusion=True`` is requested for a supported method, a sufficiently large
plan may merge adjacent operator payloads into wider ones. That rewrite is
equal to the unfused plan only to floating-point tolerance.

Shot threading uses the compiled loop in ``workflows.numba.trajectories``, while kernel
threading partitions state amplitudes or cosets, or operator columns. Serial
inner kernels use one Numba thread unless the compiled shot loop owns the outer
pool. Process-backed replay uses a one-thread child policy, so process and Numba
thread pools never nest. The compiled multi-shot path is independent of the
optional fusion rewrite.

The RNG draw stays in NumPy - a ``np.random.Generator`` cannot cross into
nopython code - so uniforms are drawn with ``rng`` and the inverse-CDF search
runs in Numba. This consumes the stream identically to ``rng.choice``, so
counts stay reproducible per engine (float-summation differences from NumPy
mean they are not bit-identical across engines - the documented contract,
see `np.py`).

Numba compiles kernels lazily on first call. This module is never imported from
``fatqat.simulator._engine``'s ``__init__``; import it explicitly.

Conventions match `np.py`: private little-endian engine indexing (engine
subsystem ``q`` has place value ``prod(dims[:q])``) and a local gate matrix
whose most-significant index digit is ``targets[0]``. The simulator maps public
subsystems before lowering; reversing an exported buffer here would apply the
mapping twice and allocate an unnecessary full-result copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import NamedTuple

import numpy as np

from ..._execution_contract import (
    RawResult,
    _InitialEvolutionState as InitialEvolutionState,
    _SimulationConfig as SimulationConfig,
    _ExecutionContext as ExecutionContext,
    _KernelCapabilities,
    _EngineCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
)
from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    LossStep,
    PutStep,
    ResetStep,
    ResolvedStep,
)
from ....noise.nb import _jump_branch_kernel
from ....result import reduce_to_counts
from ....errors import BackendValidationError
from .._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
    _resolve_kernel_scheduling,
    _resolve_python_workflow_scheduling,
    _select_execution_path,
    _validate_shot_parallelism,
    _explicit_thread_worker_ceiling,
    _adaptive_thread_worker_ceiling,
)
from .._numba_scope import _kernel_thread_workers, _thread_scope
from ..numerics.numba._apply import (
    _classify_matrix,
    _run_resolved,
    _run_resolved_sparse,
)
from ..numerics.numba._fusion import (
    _MIN_SIZE_TO_FUSE,
    _fuse_gate_channels,
    _fuse_operator_payloads,
)
from ..numerics.numba._measurement import (
    _sample_index_kernel,
    _sample_indices_kernel,
    _sv_probabilities_kernel,
    _sv_project_kernel,
    _sv_reset_kernel,
    _dm_probabilities_kernel,
    _dm_project_kernel,
)
from ..numerics.numba._preparation import (
    _MatrixStructureCache,
    _SuperoperatorCache,
    _MAX_THREADS,
    _MIN_SIZE_TO_THREAD_DM,
    _compute_apply_plan,
    _measured_layout,
)
from .numpy import (
    _NumpyOperatorEngine,
    NumpyDMEngine,
    NumpySuperopEngine,
    NumpySVEngine,
    NumpyUnitaryEngine,
)
from ..workflows.python import _PreparedPythonExecution
from ..workflows.numba.operators import (
    _operator_chunks,
    _pack_operator_plan,
    _run_operator_plan_parallel,
    _run_operator_plan_serial,
)
from ..workflows.numba.trajectories import (
    _PreparedSVTrajectories,
    _compile_dynamic_plan,
    _execute_sv_trajectories,
    _plan_compilable,
)


class _PreparedSVExecution(NamedTuple):
    """Python plan, optional compiled trajectories, and reusable SV layouts."""

    python_execution: _PreparedPythonExecution
    compiled_trajectories: _PreparedSVTrajectories | None
    apply_plans: tuple[tuple[tuple[int, ...], tuple], ...]


class NumbaSVEngine(NumpySVEngine):
    """State-vector engine with Numba-jitted numeric kernels."""

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "statevector", supports_nonunitary=True, nonunitary_is_stochastic=True
        ),
        trajectory=_TrajectoryCapabilities(classical_register=True, occupancy=True),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=False,
        ),
        supports_process_shots=True,
    )

    def plan_execution(
        self,
        plan: Sequence[ResolvedStep],
        simulation_config: SimulationConfig,
        *,
        counts_requested: bool,
        state_requested: bool,
        shots: int,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> ExecutionPolicy:
        """Choose compiled trajectories or Python sequencing with Numba kernels.

        Eligible repeated trajectories use the compiled loop with serial inner
        kernels and serial or threaded shots. Other requests use shared Python
        scheduling, including process replay and threading within kernels.
        """
        execution_path = _select_execution_path(
            plan,
            quantum_capabilities=self.capabilities.quantum,
            initial_state=initial_state,
        )
        _validate_shot_parallelism(
            simulation_config,
            execution_path=execution_path,
            counts_requested=counts_requested,
            state_requested=state_requested,
            shots=shots,
        )
        shot_request = simulation_config.shot_parallelism
        kernel_request = simulation_config.kernel_parallelism
        # Explicit processes and kernel threads use Python replay. With serial
        # shots, preserve adaptive kernel execution unless kernels are also serial.
        use_python = (
            shot_request == "processes"
            or kernel_request == "threads"
            or (shot_request == "serial" and kernel_request == "auto")
            or shots <= 1
            or execution_path
            != "per_shot"  # Plan not supported by compiled trajectories.
            or not counts_requested
            or state_requested
            or initial_state.classical.occupied is not None
            or not _plan_compilable(plan)
        )
        if use_python:
            return _resolve_python_workflow_scheduling(
                simulation_config,
                execution_path=execution_path,
                counts_requested=counts_requested,
                state_requested=state_requested,
                capabilities=self.capabilities.kernels,
                supports_process_shots=self.capabilities.supports_process_shots,
                shots=shots,
            )

        # This compiled workflow owns the shot loop and calls serial kernels.
        # Auto prefers this loop even when the worker ceiling makes it serial.
        kernels = self.capabilities.kernels
        if shot_request == "threads":
            workers = _explicit_thread_worker_ceiling(
                simulation_config.max_workers, kernels
            )
            if workers < 2:
                raise BackendValidationError(
                    "shot_parallelism='threads' requires an effective thread "
                    "capacity of at least two"
                )
            shot_strategy = "threads"
        elif shot_request == "serial":
            shot_strategy, workers = "serial", 1
        else:
            workers = _adaptive_thread_worker_ceiling(
                simulation_config.max_workers, kernels
            )
            if kernels.thread_capacity >= 2 and (workers is None or workers > 1):
                shot_strategy = "threads"
            else:
                shot_strategy, workers = "serial", 1
        return ExecutionPolicy(
            execution_path=execution_path,
            shot_strategy=shot_strategy,
            kernel_strategy="serial",
            worker_limit=workers,
            fusion=simulation_config.fusion,
            workflow="numba_trajectories",
        )

    def __init__(self, name: str = "numba-sv"):
        super().__init__(name)
        # Per-gate layout (offsets/strides/chunk count) keyed by target tuple.
        # The layout depends only on targets and the fixed system dims, so it is
        # reused across gates and shots; `initialize` clears it when dims change.
        self._apply_plans: dict[tuple[int, ...], tuple] = {}
        self._matrix_structures = _MatrixStructureCache()
        # Bind once: all consumers resolve through the numerical owner directly.
        self._resolve_structure = self._matrix_structures.resolve

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        super()._retain_step_caches(plan)
        self._matrix_structures.retain(plan)

    def configure_system(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        dims_changed = tuple(int(dim) for dim in system_dims) != self._dims
        super().configure_system(system_dims, n_clbits)
        if dims_changed:
            self._apply_plans = {}

    def _execution_scope(self, policy: ExecutionPolicy):
        return _thread_scope(policy)

    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> _PreparedSVExecution:
        python_execution = super().materialize_execution(
            plan,
            system_dims=system_dims,
            n_clbits=n_clbits,
            policy=policy,
        )
        execution_plan = python_execution.plan
        targets = {
            tuple(step.target_indices)
            for step in execution_plan
            if isinstance(step, (ApplyMatrixStep, ApplyChannelStep))
        }
        targets.update(
            (index,)
            for step in execution_plan
            if isinstance(step, ResetStep)
            for index in step.reset_indices
        )
        targets.update(
            (index,)
            for step in execution_plan
            if isinstance(step, (LossStep, PutStep))
            for index in step.target_indices
        )
        workers = _kernel_thread_workers(policy)
        self._apply_plans = {
            target: self._build_apply_plan(
                target,
                max_workers=workers,
                force_parallel=policy.kernel_strategy == "threads",
            )
            for target in targets
        }
        compiled = None
        if policy.workflow == "numba_trajectories":
            compiled = _compile_dynamic_plan(
                list(execution_plan),
                self._dims,
                self._build_apply_plan,
                self._resolve_structure,
            )
        return _PreparedSVExecution(
            python_execution, compiled, tuple(self._apply_plans.items())
        )

    def execute_local(
        self,
        context: ExecutionContext,
        payload: _PreparedSVExecution,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._apply_plans = dict(payload.apply_plans)
        if policy.workflow == "python":
            return super().execute_local(
                context, payload.python_execution, policy, initial_state=initial_state
            )
        assert policy.workflow == "numba_trajectories"
        assert payload.compiled_trajectories is not None
        with self._execution_scope(policy):
            rows = _execute_sv_trajectories(
                context, payload.compiled_trajectories, initial_state=initial_state
            )
            outcome_keys, outcome_counts = reduce_to_counts(rows)
            return RawResult(
                outcome_keys=outcome_keys,
                outcome_counts=outcome_counts,
            )

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload: _PreparedSVExecution,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> list[tuple[int, ...]]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._apply_plans = dict(payload.apply_plans)
        return super().execute_shot_batch(
            context,
            payload.python_execution,
            seed_batch,
            policy,
            initial_state=initial_state,
        )

    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply one plan step - the standard path (key-aware, cached)."""
        code, columns, values = self._resolve_structure(step)
        self._state = self._launch_resolved(
            self.state, step.matrix, step.target_indices, code, columns, values
        )

    def _apply_kraus_jump(
        self, step: ApplyChannelStep, rng: np.random.Generator
    ) -> None:
        """Sample one Kraus branch in Numba (quantum-jump unravelling).

        Each branch ``K_i |psi>`` is built by the local-apply fallback path
        from its own copy of the state (the kernels update their input buffer
        in place, so the source must not be reused), and
        `_jump_branch_kernel` does the channel-specific part: branch weights,
        the pick, the renormalization. Consumes exactly one rng draw, like the
        NumPy twin and like measurement and reset.
        """
        source = self.state
        branches = np.empty((len(step.kraus_ops), source.shape[0]), dtype=np.complex128)
        for i, kraus in enumerate(step.kraus_ops):
            branches[i] = self._apply_local(source.copy(), kraus, step.target_indices)
        self._state = _jump_branch_kernel(branches, float(rng.random()))

    def _apply_local(
        self, state: np.ndarray, matrix: np.ndarray, targets: Sequence[int]
    ) -> np.ndarray:
        """Apply a bare local matrix - the matrix-only fallback path.

        For callers with no step to carry identity or cache against, such as
        noise-channel Kraus branches. One `_classify_matrix` scan (the same
        single rule the standard path uses), then the same resolved kernels.
        """
        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
        d = matrix.shape[0]
        columns = np.empty(d, dtype=np.int64)
        values = np.empty(d, dtype=np.complex128)
        code = int(_classify_matrix(matrix, columns, values))
        return self._launch_resolved(state, matrix, targets, code, columns, values)

    def _launch_resolved(
        self,
        state: np.ndarray,
        matrix: np.ndarray,
        targets: Sequence[int],
        code: int,
        columns: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Shared kernel launch for the standard and fallback paths."""
        targets = tuple(targets)
        plan = self._apply_plans.get(targets)
        if plan is None:
            plan = self._build_apply_plan(targets)
            self._apply_plans[targets] = plan
        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
        state = np.ascontiguousarray(state, dtype=np.complex128)
        return _run_resolved(state, matrix, plan, code, columns, values)

    def _build_apply_plan(
        self,
        targets: tuple[int, ...],
        *,
        max_workers: int = _MAX_THREADS,
        force_parallel: bool = False,
    ) -> tuple:
        """Strided-block kernel layout for ``targets`` over the physical dims."""
        return _compute_apply_plan(
            self._dims,
            targets,
            max_workers=max_workers,
            force_parallel=force_parallel,
        )

    def probabilities(self) -> np.ndarray:
        return _sv_probabilities_kernel(np.ascontiguousarray(self.state))

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``shots`` flat indices: draw uniforms in NumPy, invert in Numba."""
        return _sample_indices_kernel(self.probabilities(), rng.random(shots))

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project onto it in Numba, return the flat index."""
        index = int(_sample_index_kernel(self.probabilities(), float(rng.random())))
        strides, dims = _measured_layout(self._dims, measured_subsystems)
        self._state = _sv_project_kernel(
            np.ascontiguousarray(self.state), strides, dims, index
        )
        return index

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None:
        """Reset a sampled branch to zero using Numba digit shifts and one draw."""
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        strides, dims = _measured_layout(self._dims, indices)
        self._state = _sv_reset_kernel(
            np.ascontiguousarray(self.state), strides, dims, float(rng.random())
        )


# --- density-matrix kernels ---


class _PreparedDMExecution(NamedTuple):
    """Effective Python plan and reusable layouts over the doubled DM space."""

    python_execution: _PreparedPythonExecution
    sandwich_plans: tuple[tuple[tuple[int, ...], tuple], ...]


class NumbaDMEngine(NumpyDMEngine):
    """Density-matrix engine with Numba-jitted, key-driven numeric kernels.

    Overrides the numeric kernels of `NumpyDMEngine` and materializes the
    optional gate/channel rewrite once. Inherited measurement and execution
    helpers call the Numba kernels within the resolved thread scope.
    ``reset_subsystems`` stays the inherited NumPy partial-trace channel (see
    the module docstring).

    Both a gate and a channel apply as one super-operator pass over
    ``vec(rho)`` (module docstring); `_resolve_superop` is the density-matrix
    analog of the statevector `_resolve_structure` - built and classified once
    per plan step, key-aware for gates, content-scanned for channels.
    """

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "density_matrix", supports_nonunitary=True, nonunitary_is_stochastic=False
        ),
        trajectory=_TrajectoryCapabilities(classical_register=True, occupancy=True),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=True,
        ),
        supports_process_shots=True,
    )

    def __init__(self, name: str = "numba-dm"):
        super().__init__(name)
        # Per-target super-operator layout (offsets/strides/chunk count over the
        # doubled bra+ket system), keyed by target tuple. Depends only on
        # targets and the fixed dims; `initialize` clears it when dims change.
        self._sandwich_plans: dict[tuple[int, ...], tuple] = {}
        self._superoperators = _SuperoperatorCache()
        # Bind once: DM and accumulated superoperators share numerical resolution.
        self._resolve_superop = self._superoperators.resolve

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        self._superoperators.retain(plan)

    def configure_system(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        dims_changed = tuple(int(dim) for dim in system_dims) != self._dims
        super().configure_system(system_dims, n_clbits)
        if dims_changed:
            self._sandwich_plans = {}

    def _execution_scope(self, policy: ExecutionPolicy):
        return _thread_scope(policy)

    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> _PreparedDMExecution:
        """Apply the optional gate/channel rewrite exactly once."""
        execution_plan = (
            tuple(_fuse_gate_channels(list(plan))) if policy.fusion else plan
        )
        python_execution = super().materialize_execution(
            execution_plan,
            system_dims=system_dims,
            n_clbits=n_clbits,
            policy=policy,
        )
        targets = {
            tuple(step.target_indices)
            for step in python_execution.plan
            if isinstance(step, (ApplyMatrixStep, ApplyChannelStep))
        }
        workers = _kernel_thread_workers(policy)
        self._sandwich_plans = {
            target: self._build_sandwich_plan(
                target,
                max_workers=workers,
                force_parallel=policy.kernel_strategy == "threads",
            )
            for target in targets
        }
        return _PreparedDMExecution(
            python_execution, tuple(self._sandwich_plans.items())
        )

    def execute_local(
        self,
        context: ExecutionContext,
        payload: _PreparedDMExecution,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._sandwich_plans = dict(payload.sandwich_plans)
        return super().execute_local(
            context, payload.python_execution, policy, initial_state=initial_state
        )

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload: _PreparedDMExecution,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> list[tuple[int, ...]]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._sandwich_plans = dict(payload.sandwich_plans)
        return super().execute_shot_batch(
            context,
            payload.python_execution,
            seed_batch,
            policy,
            initial_state=initial_state,
        )

    def _sandwich_plan(self, targets: tuple[int, ...]) -> tuple:
        """Super-operator apply plan for ``targets`` over the doubled dims.

        The super-target combines each gate target's ket subsystem (doubled
        index ``n + t``, stride ``size * prod(dims[:t])``) and bra subsystem
        (doubled index ``t``, stride ``prod(dims[:t])``), ket group first so
        the local index is ``ket * D + bra`` - matching ``kron(M, conj(M))``.
        """
        plan = self._sandwich_plans.get(targets)
        if plan is None:
            plan = self._build_sandwich_plan(targets)
            self._sandwich_plans[targets] = plan
        return plan

    def _build_sandwich_plan(
        self,
        targets: tuple[int, ...],
        *,
        max_workers: int = _MAX_THREADS,
        force_parallel: bool = False,
    ) -> tuple:
        n = len(self._dims)
        doubled_dims = self._dims + self._dims
        super_targets = [n + t for t in targets] + list(targets)
        return _compute_apply_plan(
            doubled_dims,
            super_targets,
            _MIN_SIZE_TO_THREAD_DM,
            max_workers=max_workers,
            force_parallel=force_parallel,
        )

    def _apply_superop(self, step: ApplyMatrixStep | ApplyChannelStep) -> None:
        """One resolved super-operator pass over ``vec(rho)``, in place.

        ``self.state`` is contiguous ``complex128`` by construction, so the
        flat view aliases it and no per-step copy of the ``4^n`` matrix is
        made. A dense-but-sparse super-operator (a resolved CSR) walks its
        nonzeros; everything else takes the structure-specialized dense /
        diagonal / permutation pass.
        """
        superop, code, columns, values, sparse = self._resolve_superop(step)
        rho = self.state
        flat = np.ascontiguousarray(rho, dtype=np.complex128).reshape(-1)
        plan = self._sandwich_plan(tuple(step.target_indices))
        if sparse is not None:
            _run_resolved_sparse(flat, sparse[0], sparse[1], sparse[2], plan)
        else:
            _run_resolved(flat, superop, plan, code, columns, values)
        self._state = flat.reshape(rho.shape)

    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply one gate - key-aware super-operator pass (cached per step)."""
        self._apply_superop(step)

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Apply the exact Kraus sum as one super-operator pass.

        Deterministic; no randomness is consumed (``rng`` is accepted for
        interface parity, like reset).
        """
        self._apply_superop(step)

    def probabilities(self) -> np.ndarray:
        return _dm_probabilities_kernel(np.ascontiguousarray(self.state))

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``shots`` flat indices: draw uniforms in NumPy, invert in Numba."""
        return _sample_indices_kernel(self.probabilities(), rng.random(shots))

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project onto it in Numba, return the flat index."""
        index = int(_sample_index_kernel(self.probabilities(), float(rng.random())))
        strides, dims = _measured_layout(self._dims, measured_subsystems)
        self._state = _dm_project_kernel(
            np.ascontiguousarray(self.state), strides, dims, index
        )
        return index


# --- operator engines ---
#
# An operator is its state twin evolved on many columns at once. A run takes the
# compiled whole-plan path; a single application falls back to the per-gate coset
# kernels over a column-batched apply plan, which is a `columns`-sized
# never-targeted subsystem prepended to the dims.


class _PreparedOperatorExecution(NamedTuple):
    """Host-side operator preparation; only the packed arrays enter Numba.
    and exports the requested operator.
    """

    packed_plan: tuple[np.ndarray, ...] | None
    scratch_rows: int
    n_columns: int
    n_chunks: int


class _NumbaOperatorRunMixin(_NumpyOperatorEngine):
    """Compiled whole-plan execution for the Numba operator engines.

    Leaves declare their capabilities and supply `_operator_row_dims` and
    `_operator_payloads`.
    """

    def plan_execution(
        self,
        plan: Sequence[ResolvedStep],
        simulation_config: SimulationConfig,
        *,
        counts_requested: bool,
        state_requested: bool,
        shots: int,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> ExecutionPolicy:
        """Choose compiled operator sequencing with serial or threaded kernels.

        Operator evolution has no independent shots. Shared scheduling resolves
        the kernel controls and worker limit for the whole-plan operator loop.
        """
        execution_path = "operator"
        _validate_shot_parallelism(
            simulation_config,
            execution_path=execution_path,
            counts_requested=counts_requested,
            state_requested=state_requested,
            shots=shots,
        )
        kernel_strategy, workers = _resolve_kernel_scheduling(
            simulation_config.kernel_parallelism,
            max_workers=simulation_config.max_workers,
            capabilities=self.capabilities.kernels,
        )
        return ExecutionPolicy(
            execution_path=execution_path,
            shot_strategy="none",
            kernel_strategy=kernel_strategy,
            worker_limit=workers,
            fusion=simulation_config.fusion,
            workflow="numba_operator",
        )

    # State-engine reuse must not override the operator measurement contract.
    probabilities = _NumpyOperatorEngine.probabilities
    collapse = _NumpyOperatorEngine.collapse

    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> _PreparedOperatorExecution:
        """Resolve, optionally fuse, and pack the operator plan once."""
        assert policy.workflow == "numba_operator"
        self.configure_system(system_dims, n_clbits)
        execution_plan = self._operator_execution_plan(plan, policy)
        self._retain_step_caches(execution_plan)
        payloads = self._operator_payloads(list(execution_plan))
        row_dims = self._operator_row_dims()
        n_columns = prod(row_dims) if row_dims else 1
        flat_size = n_columns * n_columns
        if policy.fusion and flat_size >= _MIN_SIZE_TO_FUSE:
            payloads = _fuse_operator_payloads(payloads, row_dims)
        if not payloads:
            return _PreparedOperatorExecution(None, 0, n_columns, 1)
        packed, scratch_rows = _pack_operator_plan(payloads, row_dims, n_columns)
        n_chunks = _operator_chunks(len(payloads), flat_size, n_columns, policy)
        return _PreparedOperatorExecution(packed, scratch_rows, n_columns, n_chunks)

    def _operator_execution_plan(
        self, plan: tuple[ResolvedStep, ...], policy: ExecutionPolicy
    ) -> tuple[ResolvedStep, ...]:
        del policy
        return plan

    def execute_local(
        self,
        context: ExecutionContext,
        payload: _PreparedOperatorExecution,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        """Execute one already-packed operator payload without replanning."""
        assert policy.workflow == "numba_operator"
        assert policy.shot_strategy == "none"
        assert policy.execution_path == "operator"
        self.configure_system(context.system_dims, context.n_clbits)
        with self._execution_scope(policy):
            self.initialize(
                self._dims,
                self._n_clbits,
                initial_state=initial_state,
            )
            if payload.packed_plan is not None:
                operator = np.ascontiguousarray(self.state, dtype=np.complex128)
                flat = operator.reshape(-1)
                if payload.n_chunks > 1:
                    _run_operator_plan_parallel(
                        flat,
                        payload.n_columns,
                        payload.packed_plan,
                        payload.scratch_rows,
                        payload.n_chunks,
                    )
                else:
                    _run_operator_plan_serial(
                        flat,
                        payload.n_columns,
                        payload.packed_plan,
                        payload.scratch_rows,
                    )
                self._state = flat.reshape(operator.shape)
            state = (
                self.export_state()
                if getattr(context.request, self._state_field)
                else None
            )
            return RawResult(state=state)

    def _operator_row_dims(self) -> tuple[int, ...]:
        """Subsystem dimensions the operator's row index decomposes into."""
        raise NotImplementedError

    def _operator_payloads(self, plan: list) -> list[tuple]:
        """Resolve an already-prepared ``plan`` into numeric payloads."""
        raise NotImplementedError


# Deep base list by design: the leaf adds only its plan builder and payloads.
class NumbaUnitaryEngine(  # pylint: disable=too-many-ancestors
    _NumbaOperatorRunMixin, NumbaSVEngine, NumpyUnitaryEngine
):
    """Unitary engine with Numba-jitted numeric kernels.

    ``U`` is ``size`` statevector columns over a row index that decomposes into
    the plain system dims; every step is a gate.
    """

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            # Retain the SV branch classification; non-unitary plans are rejected.
            "unitary",
            supports_nonunitary=False,
            nonunitary_is_stochastic=True,
        ),
        trajectory=_TrajectoryCapabilities(),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=True,
        ),
        supports_process_shots=False,
    )

    def __init__(self, name: str = "numba-unitary"):
        super().__init__(name)

    def _operator_row_dims(self) -> tuple[int, ...]:
        return self._dims

    def _operator_payloads(self, plan: list) -> list[tuple]:
        # A unitary plan holds only gates, so it reaches this translator as lowered.
        payloads = []
        for step in plan:
            assert isinstance(
                step, ApplyMatrixStep
            ), "unitary execution accepts only matrix steps"
            code, columns, values = self._resolve_structure(step)
            payloads.append(
                (step.matrix, code, columns, values, None, tuple(step.target_indices))
            )
        return payloads

    def _build_apply_plan(
        self,
        targets: tuple[int, ...],
        *,
        max_workers: int = _MAX_THREADS,
        force_parallel: bool = False,
    ) -> tuple:
        """Kernel layout for ``targets`` over the column-batched dims."""
        size = prod(self._dims) if self._dims else 1
        extended = (size,) + self._dims
        return _compute_apply_plan(
            extended,
            tuple(1 + t for t in targets),
            max_workers=max_workers,
            force_parallel=force_parallel,
        )

    def _launch_resolved(
        self,
        state: np.ndarray,
        matrix: np.ndarray,
        targets: Sequence[int],
        code: int,
        columns: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Run the inherited launch over the operator's flat ``(row, column)`` buffer."""
        operator = np.ascontiguousarray(state, dtype=np.complex128)
        flat = super()._launch_resolved(
            operator.reshape(-1), matrix, targets, code, columns, values
        )
        return flat.reshape(operator.shape)


# Deep base list by design: the leaf adds only its plan builder and payloads.
class NumbaSuperopEngine(  # pylint: disable=too-many-ancestors
    _NumbaOperatorRunMixin, NumbaDMEngine, NumpySuperopEngine
):
    """Super-operator engine with Numba-jitted numeric kernels.

    ``S`` is ``size**2`` density-matrix columns over a row index that decomposes
    into the doubled ``bra + ket`` dims. Every step - gate, channel, or reset -
    resolves to one super-operator on the combined ket+bra super-target.
    """

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "superop", supports_nonunitary=True, nonunitary_is_stochastic=False
        ),
        trajectory=_TrajectoryCapabilities(),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=True,
        ),
        supports_process_shots=False,
    )

    def __init__(self, name: str = "numba-superop"):
        super().__init__(name)

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        # Reset payloads resolve synthesized channels, not the ResetSteps themselves.
        reset_steps = tuple(
            self._reset_channels.resolve(index, self._dims[index])
            for step in plan
            if isinstance(step, ResetStep)
            for index in step.reset_indices
        )
        super()._retain_step_caches(plan + reset_steps)

    def _operator_row_dims(self) -> tuple[int, ...]:
        return self._dims + self._dims

    def _operator_payloads(self, plan: list) -> list[tuple]:
        n = len(self._dims)
        payloads = []
        for step in plan:
            # A reset is the Kraus channel sum_k |0><k|.
            resolved_steps = (
                [
                    self._reset_channels.resolve(index, self._dims[index])
                    for index in step.reset_indices
                ]
                if isinstance(step, ResetStep)
                else [step]
            )
            for resolved in resolved_steps:
                superop, code, columns, values, sparse = self._resolve_superop(resolved)
                targets = tuple(resolved.target_indices)
                # Ket group first, so the local index is ``ket * D + bra``.
                row_targets = tuple(n + t for t in targets) + targets
                payloads.append((superop, code, columns, values, sparse, row_targets))
        return payloads

    def _operator_execution_plan(
        self, plan: tuple[ResolvedStep, ...], policy: ExecutionPolicy
    ) -> tuple[ResolvedStep, ...]:
        if not policy.fusion:
            return plan
        return tuple(_fuse_gate_channels(list(plan)))

    def _sandwich_plan(self, targets: tuple[int, ...]) -> tuple:
        """Super-operator layout for ``targets`` over the column-batched doubled dims."""
        plan = self._sandwich_plans.get(targets)
        if plan is None:
            n = len(self._dims)
            size = prod(self._dims) if self._dims else 1
            extended = (size * size,) + self._dims + self._dims
            super_targets = [1 + n + t for t in targets] + [1 + t for t in targets]
            plan = _compute_apply_plan(extended, super_targets, _MIN_SIZE_TO_THREAD_DM)
            self._sandwich_plans[targets] = plan
        return plan
