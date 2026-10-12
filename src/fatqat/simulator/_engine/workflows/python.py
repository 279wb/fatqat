"""Shared Python state and operator sequencing for matrix-family numerical runtimes.

Functions receive numerical operations and explicit classical execution data.
Quantum storage, initialization, and export stay with the engine; no workflow
inspects or converts a quantum array. NumPy is used only for host-side random
draws and classical results, including when operations use device storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol

import numpy as np

from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    LossStep,
    MeasurementStep,
    PutStep,
    ResetStep,
    ResolvedStep,
)
from ....result import _decode_engine_indices_to_clbit_rows
from ..._execution_contract import _ExecutionContext as ExecutionContext
from ..state import ClassicalState
from ._reporting import (
    ERASURE_DIGIT,
    _apply_measurement_reporting,
    _map_physical_digit,
    _report_digit,
    _reporting_by_clbit,
)


class _PythonOperations(Protocol):
    """Numerical operations needed by Python sequencing, supplied by the engine.

    This is a structural typing contract, not a wrapper or a runtime dispatcher.
    Operations retain quantum storage in its native runtime. Sampling and
    measurement return host outcomes ready for classical decoding/feedforward;
    readiness and any required device transfers belong to the implementation.
    """

    def apply(self, step: ApplyMatrixStep) -> None: ...

    def apply_channel(
        self, step: ApplyChannelStep, rng: np.random.Generator
    ) -> None: ...

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None: ...

    def measure_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> tuple[int, ...]: ...

    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int: ...

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray: ...


class _PreparedPythonExecution(NamedTuple):
    """Effective plan and deferred measurements for Python-driven execution.

    Retains the prepared plan without copying steps or numerical arrays. Numba
    engines carry this same record alongside their runtime-specific preparation.
    """

    plan: tuple[ResolvedStep, ...]
    deferred_measurements: tuple[tuple[int, int], ...]


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None, clbits: list[int] | None
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    if not condition:
        return True
    assert clbits is not None, "feedforward requires initialized classical digits"
    return all(clbits[c] == v for c, v in condition)


def _run_single_pass(
    operations: _PythonOperations,
    plan: Sequence[ResolvedStep],
    measurements: Sequence[tuple[int, int]],
    classical: ClassicalState,
    context: ExecutionContext,
    rng: np.random.Generator,
    *,
    state_requested: bool,
) -> np.ndarray | None:
    """Evolve initialized storage once and return requested classical samples.

    The ``ResetStep`` and ``ApplyChannelStep`` branches are only reachable
    under density-matrix semantics, where both are deterministic channels;
    statevector semantics routes every plan bearing them to the dynamic
    path.
    """
    for step in plan:
        if isinstance(step, ApplyMatrixStep):
            operations.apply(step)
        elif isinstance(step, ApplyChannelStep):
            operations.apply_channel(step, rng)
        elif isinstance(step, ResetStep):
            operations.reset_subsystems(step.reset_indices, rng)

    collapsed_index: int | None = None
    if state_requested and measurements:
        collapsed_index = operations.collapse([q for q, _ in measurements], rng)

    rows = None
    if context.request.counts:
        if not measurements:
            indices = np.zeros(context.shots, dtype=int)
        elif collapsed_index is not None:
            indices = np.array([collapsed_index], dtype=int)
        else:
            indices = operations.sample_indices(context.shots, rng)
        rows = _decode_engine_indices_to_clbit_rows(
            indices, measurements, context.system_dims, context.n_clbits
        )
        if classical.clbits is not None:
            written = {clbit for _subsystem, clbit in measurements}
            for clbit, digit in enumerate(classical.clbits):
                if clbit not in written:
                    rows[:, clbit] = digit
        _apply_measurement_reporting(rows, _reporting_by_clbit(plan), rng)
    return rows


def _run_one_shot(
    operations: _PythonOperations,
    plan: Sequence[ResolvedStep],
    classical: ClassicalState,
    context: ExecutionContext,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Run one dynamic-path shot and return its final clbit snapshot.

    Shared by local and process-worker batches. Quantum state is accessed only
    through numerical operations (reset consumes rng only under SV semantics).

    Classical storage was initialized with the quantum state. Allocate
    report digits only when a measurement or condition needs them.
    """
    clbits = classical.clbits
    occupied = classical.occupied
    for step in plan:
        if clbits is None and (
            isinstance(step, MeasurementStep) or step.condition is not None
        ):
            clbits = [0] * context.n_clbits
            classical.clbits = clbits
        if isinstance(step, ApplyMatrixStep) and (
            occupied is None or all(t in occupied for t in step.target_indices)
        ):
            if _condition_matches(step.condition, clbits):
                operations.apply(step)
        elif isinstance(step, ApplyChannelStep) and (
            occupied is None or all(t in occupied for t in step.target_indices)
        ):
            if _condition_matches(step.condition, clbits):
                operations.apply_channel(step, rng)
        elif isinstance(step, LossStep):
            if _condition_matches(step.condition, clbits):
                if occupied is None:
                    occupied = set(range(len(context.system_dims)))
                    classical.occupied = occupied
                for index in step.target_indices:
                    if index in occupied and rng.random() < step.p:
                        occupied.discard(index)
                        operations.reset_subsystems([index], rng)
        elif isinstance(step, PutStep):
            if _condition_matches(step.condition, clbits) and occupied is not None:
                for index in step.target_indices:
                    if index not in occupied:
                        occupied.add(index)
                        operations.reset_subsystems([index], rng)
        elif isinstance(step, MeasurementStep):
            assert clbits is not None, "measurement requires classical digits"
            bits = operations.measure_subsystems(step.measured_indices, rng)
            confusions = step.confusions or (None,) * len(bits)
            maps = step.reported_digit_maps or (None,) * len(bits)
            # The collapse keeps the physical outcome; only its mapped,
            # optionally confused report is written to the clbits.
            for m, c, bit, reported_map, confusion in zip(
                step.measured_indices,
                step.classical_indices,
                bits,
                maps,
                confusions,
            ):
                if occupied is not None and m not in occupied:
                    clbits[c] = ERASURE_DIGIT
                else:
                    clbits[c] = _report_digit(
                        _map_physical_digit(bit, reported_map), confusion, rng
                    )
        elif isinstance(step, ResetStep):
            if _condition_matches(step.condition, clbits) and (
                occupied is None or all(t in occupied for t in step.reset_indices)
            ):
                operations.reset_subsystems(step.reset_indices, rng)
    # Unwritten report digits are zero even when no register was needed.
    return tuple(clbits) if clbits is not None else (0,) * context.n_clbits


def _run_operator_plan(
    operations: _PythonOperations,
    plan: Sequence[ResolvedStep],
    rng: np.random.Generator,
) -> None:
    """Apply an operator plan to already-initialized native storage."""
    for step in plan:
        assert not isinstance(
            step, MeasurementStep
        ), "operator execution cannot represent a measurement"
        assert (
            step.condition is None
        ), "operator execution cannot represent a feedforward condition"
        if isinstance(step, ApplyMatrixStep):
            operations.apply(step)
        elif isinstance(step, ApplyChannelStep):
            operations.apply_channel(step, rng)
        elif isinstance(step, ResetStep):
            operations.reset_subsystems(step.reset_indices, rng)
        else:
            raise TypeError(f"unknown resolved execution step {type(step).__name__}")
