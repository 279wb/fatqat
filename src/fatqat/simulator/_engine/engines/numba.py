"""Numba matrix-family engines.

`NumbaSVEngine` (statevector) and `NumbaDMEngine` (density matrix) reuse
the static capability reporting, materialization, local execution, and
``initialize`` / ``measure_subsystems`` orchestration of their NumPy twins,
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
blocks (``_NumbaOperatorRunMixin``). Those kernels keep the per-step
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

import numpy as np
from numba import njit, prange

from ..._execution_contract import (
    RawResult,
    _InitialEvolutionState as InitialEvolutionState,
    _ExecutionContext as ExecutionContext,
    _KernelCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
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
from ....noise.nb import (
    _compile_channel_table,
    _compile_readout_table,
    _jump_branch_kernel,
)
from ....result import reduce_to_counts
from .._execution_policy import _ExecutionPolicy as ExecutionPolicy
from .._numba_scope import _kernel_thread_workers, _thread_scope
from ..numerics.numba._apply import (
    _DENSE,
    _DIAGONAL,
    _PERMUTATION,
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
from ..numerics.numba._operator import (
    _apply_dense_columns,
    _apply_diagonal_columns,
    _apply_permutation_columns,
    _apply_sparse_columns,
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
from ..workflows.numba.trajectories import _run_shots_kernel

# Set by `_pack_operator_plan` for a step whose super-operator resolved to a
# CSR (`_superop_csr`); `_classify_matrix` never returns it.
_SPARSE = 3


# --- compiled whole-plan operator kernels (one parallel region) ---
#
# An operator buffer is ``(rows, columns)`` row-major and every step acts on the
# row index alone, so a block of columns is closed under the whole plan and the
# parallel region sits outside the step loop. Offsets and complement strides are
# row-space values pre-multiplied by ``columns``, so ``base + offsets[c] + lo``
# addresses the flat buffer directly.


@njit(cache=True)
def _run_operator_steps(
    state, plan, lo, hi, scratch
) -> None:  # pragma: no cover - compiled by Numba
    """Apply every step of a packed operator plan to columns ``[lo, hi)``."""
    (
        code,
        mat_flat,
        mat_ptr,
        local_dims,
        sv_flat_columns,
        sv_flat_values,
        sv_ptr,
        off_flat,
        off_ptr,
        comp_stride_flat,
        comp_dim_flat,
        comp_ptr,
        num_cosets,
        sp_indptr_flat,
        sp_indices_flat,
        sp_data_flat,
        sp_indptr_ptr,
        sp_data_ptr,
    ) = plan
    for s in range(code.shape[0]):
        local_dim = local_dims[s]
        offsets = off_flat[off_ptr[s] : off_ptr[s + 1]]
        strides = comp_stride_flat[comp_ptr[s] : comp_ptr[s + 1]]
        comp_dims = comp_dim_flat[comp_ptr[s] : comp_ptr[s + 1]]
        cosets = num_cosets[s]
        step_code = code[s]
        if step_code == _DIAGONAL:
            values = sv_flat_values[sv_ptr[s] : sv_ptr[s + 1]]
            _apply_diagonal_columns(
                state, values, offsets, strides, comp_dims, cosets, scratch, lo, hi
            )
        elif step_code == _PERMUTATION:
            cols = sv_flat_columns[sv_ptr[s] : sv_ptr[s + 1]]
            values = sv_flat_values[sv_ptr[s] : sv_ptr[s + 1]]
            _apply_permutation_columns(
                state,
                cols,
                values,
                offsets,
                strides,
                comp_dims,
                cosets,
                scratch,
                lo,
                hi,
            )
        elif step_code == _SPARSE:
            indptr = sp_indptr_flat[sp_indptr_ptr[s] : sp_indptr_ptr[s + 1]]
            indices = sp_indices_flat[sp_data_ptr[s] : sp_data_ptr[s + 1]]
            data = sp_data_flat[sp_data_ptr[s] : sp_data_ptr[s + 1]]
            _apply_sparse_columns(
                state,
                indptr,
                indices,
                data,
                offsets,
                strides,
                comp_dims,
                cosets,
                scratch,
                lo,
                hi,
            )
        else:
            matrix = mat_flat[mat_ptr[s] : mat_ptr[s + 1]].reshape(local_dim, local_dim)
            _apply_dense_columns(
                state, matrix, offsets, strides, comp_dims, cosets, scratch, lo, hi
            )


@njit(cache=True)
def _run_operator_plan_serial(
    state, n_columns, plan, scratch_rows
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Whole-plan application on one thread (no parallel region entered)."""
    scratch = np.empty(scratch_rows * n_columns, dtype=np.complex128)
    _run_operator_steps(state, plan, 0, n_columns, scratch)
    return state


@njit(cache=True, parallel=True)
def _run_operator_plan_parallel(
    state, n_columns, plan, scratch_rows, n_chunks
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Whole-plan application, split over column blocks in ONE parallel region."""
    for chunk in prange(n_chunks):  # pylint: disable=not-an-iterable
        lo = chunk * n_columns // n_chunks
        hi = (chunk + 1) * n_columns // n_chunks
        if lo < hi:
            scratch = np.empty(scratch_rows * (hi - lo), dtype=np.complex128)
            _run_operator_steps(state, plan, lo, hi, scratch)
    return state


# Minimum per-thread work before a compiled whole-plan run goes parallel.
_GRAIN_TO_THREAD_OPERATOR = 1 << 14


def _operator_chunks(
    n_steps: int, size: int, n_columns: int, policy: ExecutionPolicy
) -> int:
    """Column-block count selected by the resolved local thread policy."""
    max_workers = _kernel_thread_workers(policy)
    if policy.kernel_strategy == "serial" or max_workers <= 1:
        return 1
    if (
        policy.kernel_strategy != "threads"
        and n_steps * size < max_workers * _GRAIN_TO_THREAD_OPERATOR
    ):
        return 1
    return max(1, min(max_workers, n_columns))


def _pack_operator_plan(
    payloads: list[tuple], row_dims: tuple[int, ...], n_columns: int
) -> tuple[tuple, int]:
    """Flatten operator payloads into `_run_operator_steps`'s arrays.

    Each payload is ``(matrix, code, columns, values, sparse, row_targets)``.
    Ragged per-step data is concatenated with ``*_ptr`` index arrays; row-space
    offsets and strides are pre-multiplied by ``n_columns``.

    Returns the packed tuple and the scratch row count the widest step needs.
    """
    n_steps = len(payloads)
    code = np.empty(n_steps, dtype=np.int64)
    local_dims = np.empty(n_steps, dtype=np.int64)
    num_cosets = np.empty(n_steps, dtype=np.int64)
    mat_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    sv_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    off_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    comp_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    sp_indptr_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    sp_data_ptr = np.zeros(n_steps + 1, dtype=np.int64)

    mat_parts: list[np.ndarray] = []
    sv_column_parts: list[np.ndarray] = []
    sv_value_parts: list[np.ndarray] = []
    off_parts: list[np.ndarray] = []
    comp_stride_parts: list[np.ndarray] = []
    comp_dim_parts: list[np.ndarray] = []
    sp_indptr_parts: list[np.ndarray] = []
    sp_index_parts: list[np.ndarray] = []
    sp_data_parts: list[np.ndarray] = []
    scratch_rows = 1

    for s, (matrix, step_code, columns, values, sparse, targets) in enumerate(payloads):
        offsets, strides, dims, cosets, _ = _compute_apply_plan(row_dims, targets)
        local_dim = int(offsets.shape[0])
        off_parts.append(offsets * n_columns)
        comp_stride_parts.append(strides * n_columns)
        comp_dim_parts.append(dims)
        local_dims[s] = local_dim
        num_cosets[s] = cosets

        if sparse is not None:
            step_code = _SPARSE
        code[s] = step_code
        if step_code == _SPARSE:
            indptr, indices, data = sparse
            sp_indptr_parts.append(np.asarray(indptr, dtype=np.int64))
            sp_index_parts.append(np.asarray(indices, dtype=np.int64))
            sp_data_parts.append(np.asarray(data, dtype=np.complex128))
            # Gather block plus accumulator block.
            scratch_rows = max(scratch_rows, 2 * local_dim)
        else:
            scratch_rows = max(scratch_rows, local_dim)
            if step_code == _DENSE:
                mat_parts.append(np.asarray(matrix, dtype=np.complex128).reshape(-1))
            else:
                sv_column_parts.append(np.asarray(columns, dtype=np.int64))
                sv_value_parts.append(np.asarray(values, dtype=np.complex128))

        dense_step = step_code == _DENSE
        sparse_step = step_code == _SPARSE
        mat_ptr[s + 1] = mat_ptr[s] + (local_dim * local_dim if dense_step else 0)
        sv_ptr[s + 1] = sv_ptr[s] + (0 if dense_step or sparse_step else local_dim)
        off_ptr[s + 1] = off_ptr[s] + local_dim
        comp_ptr[s + 1] = comp_ptr[s] + len(dims)
        sp_indptr_ptr[s + 1] = sp_indptr_ptr[s] + (
            len(sp_indptr_parts[-1]) if sparse_step else 0
        )
        sp_data_ptr[s + 1] = sp_data_ptr[s] + (
            len(sp_data_parts[-1]) if sparse_step else 0
        )

    def _join(parts: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

    packed = (
        code,
        _join(mat_parts, np.complex128),
        mat_ptr,
        local_dims,
        _join(sv_column_parts, np.int64),
        _join(sv_value_parts, np.complex128),
        sv_ptr,
        _join(off_parts, np.int64),
        off_ptr,
        _join(comp_stride_parts, np.int64),
        _join(comp_dim_parts, np.int64),
        comp_ptr,
        num_cosets,
        _join(sp_indptr_parts, np.int64),
        _join(sp_index_parts, np.int64),
        _join(sp_data_parts, np.complex128),
        sp_indptr_ptr,
        sp_data_ptr,
    )
    return packed, scratch_rows


def _plan_compilable(plan: Sequence[ResolvedStep]) -> bool:
    """Whether the compiled multi-shot kernel understands every plan step.

    The kernel compiles every step type the matrix family lowers today - matrix,
    channel, measurement (including readout confusion), and reset. It does not yet
    encode non-identity physical-to-reported measurement maps. An unsupported
    step or map must not reach ``_compile_dynamic_plan``; the caller falls back
    to the inherited NumPy per-shot path instead, which executes every step type
    correctly at NumPy speed.
    """
    # pylint: disable-next=fixme
    # TODO: Compile non-identity ``reported_digit_maps`` into the measurement
    # table and apply them after physical collapse but before readout confusion.
    # Until then, keep routing such plans through the semantics-complete NumPy
    # per-shot executor rather than silently reporting the physical digit.
    return all(
        isinstance(
            step, (ApplyMatrixStep, ApplyChannelStep, MeasurementStep, ResetStep)
        )
        and not (
            isinstance(step, MeasurementStep)
            and step.reported_digit_maps is not None
            and any(
                reported_map != tuple(range(len(reported_map)))
                for reported_map in step.reported_digit_maps
            )
        )
        for step in plan
    )


class NumbaSVEngine(NumpySVEngine):
    """State-vector engine with Numba-jitted numeric kernels."""

    _quantum_capabilities = _QuantumCapabilities(
        "statevector", supports_nonunitary=True, nonunitary_is_stochastic=True
    )
    _trajectory_capabilities = _TrajectoryCapabilities(
        classical_register=True, occupancy=True
    )
    _supports_process_shots = True
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=True,
        thread_capacity=_MAX_THREADS,
        supports_fusion=False,
    )

    def compiled_multi_shot_compatible(self, plan: Sequence[ResolvedStep]) -> bool:
        """Return whether the compiled outer loop can encode this exact plan."""
        return _plan_compilable(plan)

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
    ):
        base = super().materialize_execution(
            plan,
            system_dims=system_dims,
            n_clbits=n_clbits,
            policy=policy,
        )
        execution_plan = base[0]
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
        if policy.use_compiled_multi_shot_kernel:
            compiled = self._compile_dynamic_plan(list(execution_plan))
        return (*base, compiled, tuple(self._apply_plans.items()))

    def execute_local(
        self,
        context: ExecutionContext,
        payload,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._apply_plans = dict(payload[3])
        if not policy.use_compiled_multi_shot_kernel:
            return super().execute_local(
                context, payload, policy, initial_state=initial_state
            )
        with self._execution_scope(policy):
            return self._run_compiled_multi_shot(
                context, payload[2], initial_state=initial_state
            )

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> list[tuple[int, ...]]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._apply_plans = dict(payload[3])
        return super().execute_shot_batch(
            context, payload, seed_batch, policy, initial_state=initial_state
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

    def _run_compiled_multi_shot(
        self,
        context: ExecutionContext,
        compiled,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        """Execute a materialized counts plan in one compiled multi-shot call."""
        plan_arrays, max_draws = compiled
        shots = context.shots
        uniforms = np.empty(shots * max_draws, dtype=np.float64)
        for s, seed_sequence in enumerate(
            np.random.SeedSequence(context.seed).spawn(shots)
        ):
            uniforms[s * max_draws : (s + 1) * max_draws] = np.random.default_rng(
                seed_sequence
            ).random(max_draws)

        state_size = prod(self._dims) if self._dims else 1
        custom_start = initial_state.quantum is not None
        # The custom template is read-only: each shot takes the private copy it
        # evolves. `ascontiguousarray` therefore aliases the validated common
        # case and copies only when layout or dtype requires it. The default
        # path passes no O(D) template at all.
        start = (
            np.ascontiguousarray(initial_state.quantum, dtype=np.complex128).reshape(
                state_size
            )
            if custom_start
            else np.empty(0, dtype=np.complex128)
        )
        clbits = initial_state.classical.clbits
        clbits_start = np.asarray(clbits if clbits is not None else (), dtype=np.int64)
        rows = _run_shots_kernel(
            *plan_arrays,
            start,
            state_size,
            custom_start,
            clbits_start,
            self._n_clbits,
            shots,
            uniforms,
            max_draws,
        )
        outcome_keys, outcome_counts = reduce_to_counts(rows)
        return RawResult(
            outcome_keys=outcome_keys,
            outcome_counts=outcome_counts,
            state=None,
        )

    def _compile_dynamic_plan(self, plan: list) -> tuple[tuple, int]:
        """Flatten a plan into typed arrays for `_run_shots_kernel`.

        Reuses the cached per-target apply layout; measurement/reset carry their
        subsystem strides and dims for in-kernel collapse and reset-shift, while
        the two noise payloads go to ``noise.nb``: a channel hands its Kraus
        stack plus that same layout to `_compile_channel_table`, and each
        measurement hands its confusion tuple to `_compile_readout_table`, whose
        pointers are indexed by the measured-subsystem position built here.

        Returns the kernel's positional plan arrays and the per-shot uniform-draw
        budget: one per measurement, one more per confusion-bearing measured
        subsystem, and one per reset and channel - the upper bound on RNG draws
        (a conditioned reset or channel that never fires simply draws less).
        """
        dims = self._dims
        strides = [prod(dims[:q]) for q in range(len(dims))]

        step_kind: list[int] = []
        step_data: list[int] = []
        step_cond_ptr: list[int] = []
        step_cond_len: list[int] = []
        cond_clbit: list[int] = []
        cond_value: list[int] = []
        ap_mat_ptr, ap_dim, ap_off_ptr, ap_comp_ptr, ap_comp_len = [], [], [], [], []
        ap_code = []
        mat_flat, off_flat, comp_stride_flat, comp_dim_flat = [], [], [], []
        col_flat: list[int] = []
        val_flat: list[complex] = []
        me_ptr, me_len, me_classical, me_stride, me_dim = [], [], [], [], []
        rs_ptr, rs_len, rs_stride, rs_dim = [], [], [], []
        # Noise payloads, flattened in one pass below by the noise package:
        # (kraus_ops, offsets, comp_strides, comp_dims) per channel occurrence,
        # and (num_subsystems, confusions) per measurement - the latter in
        # measurement order, which is the order me_* is filled in.
        channel_entries: list[tuple] = []
        readout_entries: list[tuple] = []
        num_measurements = 0
        num_resets = 0

        def add_condition(condition):
            step_cond_ptr.append(len(cond_clbit))
            step_cond_len.append(0 if condition is None else len(condition))
            for clbit, value in condition or ():
                cond_clbit.append(clbit)
                cond_value.append(value)

        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                offsets, comp_strides, comp_dims, _, _ = self._build_apply_plan(
                    step.target_indices
                )
                step_kind.append(0)
                step_data.append(len(ap_dim))
                add_condition(step.condition)
                ap_dim.append(offsets.shape[0])
                ap_mat_ptr.append(len(mat_flat))
                matrix = np.ascontiguousarray(step.matrix, dtype=np.complex128)
                mat_flat.extend(matrix.ravel().tolist())
                # Structure resolved once at compile time (declared key or a
                # single content scan) instead of per gate per shot in-kernel.
                # col/val ride the same per-gate pointer as the offsets; a
                # dense gate stores zero padding the kernel never reads.
                code, columns, values = self._resolve_structure(step)
                ap_code.append(code)
                local_dim = offsets.shape[0]
                if code == _DENSE:
                    col_flat.extend([0] * local_dim)
                    val_flat.extend([0j] * local_dim)
                else:
                    col_flat.extend(int(c) for c in columns)
                    val_flat.extend(complex(v) for v in values)
                ap_off_ptr.append(len(off_flat))
                off_flat.extend(int(o) for o in offsets)
                ap_comp_ptr.append(len(comp_stride_flat))
                ap_comp_len.append(comp_strides.shape[0])
                comp_stride_flat.extend(int(s) for s in comp_strides)
                comp_dim_flat.extend(int(d) for d in comp_dims)
            elif isinstance(step, MeasurementStep):
                step_kind.append(1)
                step_data.append(len(me_len))
                add_condition(None)
                me_ptr.append(len(me_stride))
                me_len.append(len(step.measured_indices))
                for q in step.measured_indices:
                    me_stride.append(strides[q])
                    me_dim.append(dims[q])
                me_classical.extend(step.classical_indices)
                readout_entries.append((len(step.measured_indices), step.confusions))
                num_measurements += 1
            elif isinstance(step, ApplyChannelStep):
                offsets, comp_strides, comp_dims, _, _ = self._build_apply_plan(
                    step.target_indices
                )
                step_kind.append(3)
                step_data.append(len(channel_entries))
                add_condition(step.condition)
                channel_entries.append(
                    (step.kraus_ops, offsets, comp_strides, comp_dims)
                )
            else:  # ResetStep
                step_kind.append(2)
                step_data.append(len(rs_len))
                add_condition(step.condition)
                rs_ptr.append(len(rs_stride))
                rs_len.append(len(step.reset_indices))
                for q in step.reset_indices:
                    rs_stride.append(strides[q])
                    rs_dim.append(dims[q])
                num_resets += 1

        def i64(values):
            return np.asarray(values, dtype=np.int64)

        me_conf_ptr, conf_flat = _compile_readout_table(readout_entries)
        plan_arrays = (
            i64(step_kind),
            i64(step_data),
            i64(step_cond_ptr),
            i64(step_cond_len),
            i64(cond_clbit),
            i64(cond_value),
            i64(ap_mat_ptr),
            i64(ap_dim),
            i64(ap_off_ptr),
            i64(ap_comp_ptr),
            i64(ap_comp_len),
            i64(ap_code),
            np.asarray(mat_flat, dtype=np.complex128),
            i64(off_flat),
            i64(col_flat),
            np.asarray(val_flat, dtype=np.complex128),
            i64(comp_stride_flat),
            i64(comp_dim_flat),
            i64(me_ptr),
            i64(me_len),
            i64(me_classical),
            i64(me_stride),
            i64(me_dim),
            me_conf_ptr,
            conf_flat,
            i64(rs_ptr),
            i64(rs_len),
            i64(rs_stride),
            i64(rs_dim),
            *_compile_channel_table(channel_entries),
        )
        num_confusions = int(np.count_nonzero(me_conf_ptr >= 0))
        max_draws = (
            num_measurements + num_confusions + num_resets + len(channel_entries)
        )
        return plan_arrays, max_draws

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
        strides, dims = self._measured_layout(measured_subsystems)
        self._state = _sv_project_kernel(
            np.ascontiguousarray(self.state), strides, dims, index
        )
        return index

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None:
        """Reset a sampled branch to zero using Numba digit shifts and one draw."""
        assert len(indices) >= 1, "reset_subsystems requires at least one index"
        strides, dims = self._measured_layout(indices)
        self._state = _sv_reset_kernel(
            np.ascontiguousarray(self.state), strides, dims, float(rng.random())
        )

    def _measured_layout(
        self, subsystems: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat strides and dimensions of the measured subsystems for the kernel."""
        return _measured_layout(self._dims, subsystems)


# --- density-matrix kernels ---


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

    _quantum_capabilities = _QuantumCapabilities(
        "density_matrix", supports_nonunitary=True, nonunitary_is_stochastic=False
    )
    _trajectory_capabilities = _TrajectoryCapabilities(
        classical_register=True, occupancy=True
    )
    _supports_process_shots = True
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=True, thread_capacity=_MAX_THREADS, supports_fusion=True
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
    ):
        """Apply the optional gate/channel rewrite exactly once."""
        execution_plan = (
            tuple(_fuse_gate_channels(list(plan))) if policy.fusion else plan
        )
        execution_plan, measurements = super().materialize_execution(
            execution_plan,
            system_dims=system_dims,
            n_clbits=n_clbits,
            policy=policy,
        )
        targets = {
            tuple(step.target_indices)
            for step in execution_plan
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
        return (
            execution_plan,
            measurements,
            tuple(self._sandwich_plans.items()),
        )

    def execute_local(
        self,
        context: ExecutionContext,
        payload,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._sandwich_plans = dict(payload[2])
        return super().execute_local(
            context, payload, policy, initial_state=initial_state
        )

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> list[tuple[int, ...]]:
        self.configure_system(context.system_dims, context.n_clbits)
        self._sandwich_plans = dict(payload[2])
        return super().execute_shot_batch(
            context, payload, seed_batch, policy, initial_state=initial_state
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


class _NumbaOperatorRunMixin(_NumpyOperatorEngine):
    """Compiled whole-plan execution for the Numba operator engines.

    Leaves declare their capabilities and supply `_operator_row_dims` and
    `_operator_payloads`.
    """

    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ):
        """Resolve, optionally fuse, and pack the operator plan once."""
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
            return None, 0, n_columns, 1
        packed, scratch_rows = _pack_operator_plan(payloads, row_dims, n_columns)
        n_chunks = _operator_chunks(len(payloads), flat_size, n_columns, policy)
        return packed, scratch_rows, n_columns, n_chunks

    def _operator_execution_plan(
        self, plan: tuple[ResolvedStep, ...], policy: ExecutionPolicy
    ) -> tuple[ResolvedStep, ...]:
        del policy
        return plan

    def execute_local(
        self,
        context: ExecutionContext,
        payload,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        """Execute one already-packed operator payload without replanning."""
        assert policy.shot_strategy == "none"
        assert policy.execution_path == "operator"
        self.configure_system(context.system_dims, context.n_clbits)
        packed, scratch_rows, n_columns, n_chunks = payload
        with self._execution_scope(policy):
            self.initialize(
                self._dims,
                self._n_clbits,
                initial_state=initial_state,
            )
            if packed is not None:
                operator = np.ascontiguousarray(self.state, dtype=np.complex128)
                flat = operator.reshape(-1)
                if n_chunks > 1:
                    _run_operator_plan_parallel(
                        flat, n_columns, packed, scratch_rows, n_chunks
                    )
                else:
                    _run_operator_plan_serial(flat, n_columns, packed, scratch_rows)
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

    _quantum_capabilities = _QuantumCapabilities(
        # Retain the SV branch classification; non-unitary plans are rejected.
        "unitary",
        supports_nonunitary=False,
        nonunitary_is_stochastic=True,
    )
    _trajectory_capabilities = _TrajectoryCapabilities()
    _supports_process_shots = False
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=True, thread_capacity=_MAX_THREADS, supports_fusion=True
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

    _quantum_capabilities = _QuantumCapabilities(
        "superop", supports_nonunitary=True, nonunitary_is_stochastic=False
    )
    _trajectory_capabilities = _TrajectoryCapabilities()
    _supports_process_shots = False
    _kernel_capabilities = _KernelCapabilities(
        supports_kernel_threads=True, thread_capacity=_MAX_THREADS, supports_fusion=True
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
