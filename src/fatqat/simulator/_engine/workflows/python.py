"""Shared Python state sequencing for matrix-family numerical runtimes.

Functions bind directly to the matrix engines, which supply numerical operations
and evolving state. The workflow owns step sequencing, classical updates, and
random-stream coordination for single-evolution and per-shot execution.
"""

from __future__ import annotations

from collections.abc import Sequence

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
from ....result import _decode_engine_indices_to_clbit_rows, reduce_to_counts
from ..._execution_contract import (
    _ResultRequest as ResultRequest,
    RawResult,
    _InitialEvolutionState as InitialEvolutionState,
    _ExecutionContext as ExecutionContext,
)
from ._reporting import (
    ERASURE_DIGIT,
    _apply_measurement_reporting,
    _map_physical_digit,
    _report_digit,
    _reporting_by_clbit,
)


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None, clbits: list[int] | None
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    if not condition:
        return True
    assert clbits is not None, "feedforward requires initialized classical digits"
    return all(clbits[c] == v for c, v in condition)


def _run_single_pass(
    engine,
    plan: Sequence[ResolvedStep],
    measurements: Sequence[tuple[int, int]],
    shots: int,
    rng: np.random.Generator,
    request: ResultRequest,
    initial_state: InitialEvolutionState[np.ndarray],
) -> RawResult[np.ndarray]:
    """Evolve once, optionally sample counts, optionally export the state.

    The ``ResetStep`` and ``ApplyChannelStep`` branches are only reachable
    under density-matrix semantics, where both are deterministic channels;
    statevector semantics routes every plan bearing them to the dynamic
    path.
    """
    state_requested = getattr(request, engine._state_field)
    engine.initialize(
        engine._dims,
        engine._n_clbits,
        initial_state=initial_state,
    )
    for step in plan:
        if isinstance(step, ApplyMatrixStep):
            engine.apply(step)
        elif isinstance(step, ApplyChannelStep):
            engine.apply_channel(step, rng)
        elif isinstance(step, ResetStep):
            engine.reset_subsystems(step.reset_indices, rng)

    collapsed_index: int | None = None
    if state_requested and measurements:
        collapsed_index = engine.collapse([q for q, _ in measurements], rng)

    outcome_keys = outcome_counts = state = None
    if request.counts:
        if not measurements:
            indices = np.zeros(shots, dtype=int)
        elif collapsed_index is not None:
            indices = np.array([collapsed_index], dtype=int)
        else:
            indices = engine.sample_indices(shots, rng)
        rows = _decode_engine_indices_to_clbit_rows(
            indices, measurements, engine._dims, engine._n_clbits
        )
        if initial_state.classical.clbits is not None:
            written = {clbit for _subsystem, clbit in measurements}
            for clbit, digit in enumerate(initial_state.classical.clbits):
                if clbit not in written:
                    rows[:, clbit] = digit
        _apply_measurement_reporting(rows, _reporting_by_clbit(plan), rng)
        outcome_keys, outcome_counts = reduce_to_counts(rows)
    if state_requested:
        state = engine.export_state()
    return RawResult(
        outcome_keys=outcome_keys, outcome_counts=outcome_counts, state=state
    )


def _run_per_shot(
    engine,
    plan: Sequence[ResolvedStep],
    context: ExecutionContext,
    *,
    initial_state: InitialEvolutionState[np.ndarray],
) -> RawResult[np.ndarray]:
    """Replay dynamic trajectories locally under an already-resolved policy."""
    request = context.request
    state_requested = getattr(request, engine._state_field)
    n_iters = context.shots if request.counts else (1 if state_requested else 0)
    snapshots = engine._run_shot_batch(
        plan,
        np.random.SeedSequence(context.seed).spawn(n_iters),
        initial_state,
    )

    outcome_keys = outcome_counts = state = None
    if request.counts:
        rows = np.asarray(snapshots, dtype=int).reshape(
            (len(snapshots), engine._n_clbits)
        )
        outcome_keys, outcome_counts = reduce_to_counts(rows)
    if state_requested:
        state = engine.export_state()
    return RawResult(
        outcome_keys=outcome_keys, outcome_counts=outcome_counts, state=state
    )


def _run_shot_batch(
    engine,
    plan: Sequence[ResolvedStep],
    seed_sequences: Sequence[np.random.SeedSequence],
    initial_state: InitialEvolutionState[np.ndarray],
) -> list[tuple[int, ...]]:
    snapshots = []
    for seed_sequence in seed_sequences:
        engine.initialize(
            engine._dims,
            engine._n_clbits,
            initial_state=initial_state,
        )
        snapshots.append(
            engine._run_one_shot(plan, np.random.default_rng(seed_sequence))
        )
    return snapshots


def _run_one_shot(
    engine,
    plan: Sequence[ResolvedStep],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Run one dynamic-path shot and return its final clbit snapshot.

    Shared by the serial path and `parallel.py`'s worker batches; it touches
    the state only through the interface methods (reset consumes rng only
    under statevector semantics).

    Classical storage was initialized with the quantum state. Allocate
    report digits only when a measurement or condition needs them.
    """
    assert engine._evolution_state is not None
    classical = engine._evolution_state.classical
    clbits = classical.clbits
    occupied = classical.occupied
    for step in plan:
        if clbits is None and (
            isinstance(step, MeasurementStep) or step.condition is not None
        ):
            clbits = [0] * engine._n_clbits
            classical.clbits = clbits
        if isinstance(step, ApplyMatrixStep) and (
            occupied is None or all(t in occupied for t in step.target_indices)
        ):
            if _condition_matches(step.condition, clbits):
                engine.apply(step)
        elif isinstance(step, ApplyChannelStep) and (
            occupied is None or all(t in occupied for t in step.target_indices)
        ):
            if _condition_matches(step.condition, clbits):
                engine.apply_channel(step, rng)
        elif isinstance(step, LossStep):
            if _condition_matches(step.condition, clbits):
                if occupied is None:
                    occupied = set(range(len(engine._dims)))
                    classical.occupied = occupied
                for index in step.target_indices:
                    if index in occupied and rng.random() < step.p:
                        occupied.discard(index)
                        engine.reset_subsystems([index], rng)
        elif isinstance(step, PutStep):
            if _condition_matches(step.condition, clbits) and occupied is not None:
                for index in step.target_indices:
                    if index not in occupied:
                        occupied.add(index)
                        engine.reset_subsystems([index], rng)
        elif isinstance(step, MeasurementStep):
            assert clbits is not None, "measurement requires classical digits"
            bits = engine.measure_subsystems(step.measured_indices, rng)
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
                engine.reset_subsystems(step.reset_indices, rng)
    # Unwritten report digits are zero even when no register was needed.
    return tuple(clbits) if clbits is not None else (0,) * engine._n_clbits
