import numpy as np
import pytest

from fatqat.program import Program
from fatqat.simulator import Simulator
import fatqat.operations as ops
from fatqat.simulator._engine.state import EvolutionState
from fatqat.simulator._engine.numerics.numpy._measurement import (
    _statevector_probabilities,
    _project_statevector,
    _sample_basis_indices,
)
from fatqat.simulator._engine.workflows.python import (
    _MeasurementKernels,
    _measure_subsystems,
)

from fatqat.simulator._engine.engines.numpy import NumpySVEngine
from fatqat._backends.steps import ApplyMatrixStep

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


@pytest.fixture(name="sv_engine_class", params=["numpy", "numba"])
def fixture_sv_engine_class(request):
    if request.param == "numba":
        pytest.importorskip("numba")
        from fatqat.simulator._engine.engines.numba import NumbaSVEngine

        return NumbaSVEngine
    return NumpySVEngine


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_repeated_measurement_reports_the_same_outcome(runtime):
    program = Program(1, 2)
    program.add(ops.H, 0)
    program.measure(0, 0)
    program.measure(0, 1)
    counts = (
        Simulator("SV", runtime=runtime)
        .run(
            program,
            shots=32,
            simulation_config={"seed": 0, "shot_parallelism": "serial"},
        )
        .result()
        .get_counts()
    )
    assert sum(counts.values()) == 32
    assert set(counts) <= {"00", "11"}


def test_reset_qubit_from_one_returns_zero():
    eng = NumpySVEngine()
    eng.initialize((2,))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))  # |1>
    eng.reset_subsystem(0, np.random.default_rng(0))
    assert np.allclose(eng.export_state(), np.array([1, 0], dtype=complex))


def test_reset_qubit_on_entangled_pair_conditions_the_partner(sv_engine_class):
    # Bell pair (|00>+|11>)/sqrt2; reset qubit 0 -> partner is |0> or |1>, 50/50.
    outcomes = []
    for s in range(200):
        eng = sv_engine_class()
        eng.initialize((2, 2))
        eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
        cx = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )
        eng.apply(ApplyMatrixStep(matrix=cx, target_indices=(0, 1)))
        eng.reset_subsystem(0, np.random.default_rng(s))
        st = eng.export_state()
        # qubit 0 must be |0>: only indices with bit0 == 0 are allowed.
        # Amplitude is on index 0 (partner 0) or 2 (partner 1).
        nz = np.flatnonzero(np.round(np.abs(st), 6))
        assert nz.size == 1 and nz[0] in (0, 2)
        outcomes.append(int(nz[0] == 2))  # partner==1
    frac = sum(outcomes) / len(outcomes)
    assert 0.35 < frac < 0.65


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_measurement_preserves_requested_target_order(runtime):
    program = Program(3, 2)
    program.add(ops.X, 2)
    program.measure((2, 0), (0, 1))
    counts = (
        Simulator("SV", runtime=runtime)
        .run(
            program,
            shots=1,
            simulation_config={"seed": 0},
        )
        .result()
        .get_counts_as_tuples()
    )
    assert counts == {(1, 0): 1}


def test_measure_subsystems_consumes_one_rng_draw_for_grouped_event():
    # Seed compatibility deliberately constrains grouped measurement to one draw.
    state = EvolutionState(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
    rng_grouped = np.random.default_rng(11)
    kernels = _MeasurementKernels(
        _statevector_probabilities, _project_statevector, _sample_basis_indices
    )
    _measure_subsystems(state, kernels, (2, 2), (0, 1), rng_grouped)
    rng_one_draw = np.random.default_rng(11)
    rng_one_draw.choice(4, p=np.array([0.5, 0.0, 0.0, 0.5]))
    assert rng_grouped.random() == rng_one_draw.random()


def test_reset_subsystems_resets_all_targets_with_one_grouped_collapse(sv_engine_class):
    eng = sv_engine_class()
    eng.initialize((2, 3, 2))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    shift = np.roll(np.eye(3, dtype=complex), 2, axis=0)
    eng.apply(ApplyMatrixStep(matrix=shift, target_indices=(1,)))
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(2,)))
    rng = np.random.default_rng(0)

    eng.reset_subsystems((1, 0), rng)

    expected = np.zeros(12, dtype=complex)
    expected[[0, 6]] = 1 / np.sqrt(2)
    assert np.allclose(eng.export_state(), expected)
    reference_rng = np.random.default_rng(0)
    reference_rng.random()
    assert rng.random() == reference_rng.random()
