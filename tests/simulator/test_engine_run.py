import numpy as np
import pytest

from fatqat.simulator._execution_contract import (
    _ResultRequest,
    _SimulationConfig,
    _ExecutionContext,
    _InitialEvolutionState,
    _InitialClassicalState,
)
from fatqat._backends.steps import (
    ApplyMatrixStep,
    LossStep,
    MeasurementStep,
    PutStep,
    ResetStep,
)
from fatqat.simulator._engine.engines.numpy import NumpySVEngine, NumpyUnitaryEngine
from fatqat.errors import BackendValidationError
from fatqat.simulator._engine._execution_policy import _ExecutionPolicy

_SERIAL = _ExecutionPolicy(
    execution_path="single_pass",
    shot_strategy="none",
    kernel_strategy="serial",
    worker_limit=1,
    fusion=False,
)


def _context(
    *,
    n_clbits,
    shots,
    request,
    seed=0,
):
    return _ExecutionContext(
        request=request,
        system_dims=(2,),
        n_clbits=n_clbits,
        shots=shots,
        seed=seed,
    )


def _run(
    engine,
    plan,
    context,
    *,
    policy=_SERIAL,
    initial_state=None,
):
    payload = engine.materialize_execution(
        tuple(plan),
        system_dims=context.system_dims,
        n_clbits=context.n_clbits,
        policy=policy,
    )
    return engine.execute_in_process(
        context,
        payload,
        policy,
        initial_state=(
            initial_state if initial_state is not None else _InitialEvolutionState()
        ),
    )


@pytest.mark.parametrize("density_matrix", [False, True])
def test_execution_survives_building_another_system(density_matrix):
    pytest.importorskip("numba")
    from fatqat.simulator._engine.engines.numba import NumbaSVEngine, NumbaDMEngine

    engine = NumbaDMEngine() if density_matrix else NumbaSVEngine()
    plan = (ApplyMatrixStep(np.array([[0, 1], [1, 0]], dtype=complex), (0,)),)
    contexts = [
        _ExecutionContext(
            request=_ResultRequest(counts=False, final_state=True),
            system_dims=dims,
            n_clbits=0,
            shots=1,
            seed=0,
        )
        for dims in ((2,), (2, 3))
    ]
    executions = [
        engine.materialize_execution(
            plan, system_dims=context.system_dims, n_clbits=0, policy=_SERIAL
        )
        for context in contexts
    ]
    for index in (0, 1, 0):
        context = contexts[index]
        result = engine.execute_in_process(
            context, executions[index], _SERIAL, initial_state=_InitialEvolutionState()
        )
        expected = np.zeros(np.prod(context.system_dims), dtype=complex)
        expected[1] = 1
        if density_matrix:
            expected = np.outer(expected, expected.conj())
        np.testing.assert_allclose(result.state, expected)


@pytest.mark.parametrize("with_loss", [False, True])
def test_numba_trajectory_execution_handles_supported_and_fallback_plans(with_loss):
    from fatqat.simulator._engine.engines.numba import NumbaSVEngine

    plan = (ResetStep((0,)),)
    if with_loss:
        plan += (LossStep((0,), p=1.0),)
    plan += (MeasurementStep((0,), (0,)),)
    engine = NumbaSVEngine()
    initial = _InitialEvolutionState()
    context = _context(
        n_clbits=1,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    result = engine.execute(
        plan,
        context=context,
        simulation_config=_SimulationConfig(
            shot_parallelism="serial", kernel_parallelism="serial"
        ),
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[2 if with_loss else 0]]
    assert result.outcome_counts.tolist() == [4]


def test_engine_fast_counts_returns_arrays():
    engine = NumpySVEngine()
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    plan = [ApplyMatrixStep(x, (0,)), MeasurementStep((0,), (0,))]
    context = _context(
        n_clbits=1,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )

    result = _run(engine, plan, context)

    assert result.state is None
    assert result.outcome_keys.tolist() == [[1]]
    assert result.outcome_counts.tolist() == [4]


def test_engine_fast_counts_and_state_share_collapse_event():
    engine = NumpySVEngine()
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    plan = [ApplyMatrixStep(h, (0,)), MeasurementStep((0,), (0,))]
    context = _context(
        n_clbits=1,
        shots=1,
        seed=2026,
        request=_ResultRequest(counts=True, final_state=True),
    )

    result = _run(engine, plan, context)

    measured = int(result.outcome_keys[0, 0])
    assert result.outcome_counts.tolist() == [1]
    assert np.isclose(abs(result.state[measured]), 1.0)
    assert np.count_nonzero(np.abs(result.state) > 1e-12) == 1


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize(
    ("occupied", "reported"),
    [(None, 1), (frozenset(), 0), (frozenset({0}), 1)],
)
def test_occupancy_loss_reload_and_feedback_start_fresh(runtime, occupied, reported):
    if runtime == "numba":
        from fatqat.simulator._engine.engines.numba import NumbaSVEngine

        engine = NumbaSVEngine()
    else:
        engine = NumpySVEngine()
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    plan = [
        ApplyMatrixStep(x, (0,)),
        PutStep((0,)),  # Put preserves loaded atoms and initializes empty sites.
        MeasurementStep((0,), (0,)),
        LossStep((0,), p=1.0),
        MeasurementStep((0,), (1,)),
        PutStep((0,)),
        ApplyMatrixStep(x, (0,), condition=((0, 1),)),
        MeasurementStep((0,), (2,)),
    ]
    context = _context(
        n_clbits=3,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    initial = _InitialEvolutionState(
        classical=_InitialClassicalState(occupied=occupied)
    )
    policy = _ExecutionPolicy(
        execution_path="per_shot",
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
        fusion=False,
    )

    result = _run(engine, plan, context, policy=policy, initial_state=initial)

    assert result.outcome_keys.tolist() == [[reported, 2, reported]]
    assert result.outcome_counts.tolist() == [4]

    # Reusing the engine restores initial occupancy and clears report digits.
    reused = _run(engine, plan[:3], context, policy=policy, initial_state=initial)
    assert reused.outcome_keys.tolist() == [[reported, 0, 0]]
    assert reused.outcome_counts.tolist() == [4]
    assert result.outcome_keys.tolist() == [[reported, 2, reported]]

    # Unused declared registers still report zero without evolving any digits.
    unwritten = _run(
        engine,
        [plan[0], LossStep((0,), p=0.0)],
        context,
        policy=policy,
        initial_state=initial,
    )
    assert unwritten.outcome_keys.tolist() == [[0, 0, 0]]
    assert unwritten.outcome_counts.tolist() == [4]


@pytest.mark.parametrize("mode", ["numpy", "numba", "compiled", "processes"])
def test_initial_register_drives_feedback_and_is_fresh_per_shot(mode):
    if mode == "numpy":
        engine = NumpySVEngine()
    else:
        from fatqat.simulator._engine.engines.numba import NumbaSVEngine

        engine = NumbaSVEngine()
    initial = _InitialEvolutionState(
        quantum=np.array([0, 1], dtype=complex),
        classical=_InitialClassicalState(clbits=(1, 2)),
    )
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    plan = [ApplyMatrixStep(x, (0,), condition=((0, 1),)), MeasurementStep((0,), (0,))]
    context = _context(
        n_clbits=2,
        shots=8,
        request=_ResultRequest(counts=True, final_state=False),
    )
    result = engine.execute(
        tuple(plan),
        context=context,
        simulation_config=_SimulationConfig(
            shot_parallelism="processes" if mode == "processes" else "serial",
            kernel_parallelism="serial" if mode != "numba" else "auto",
            max_workers=2 if mode == "processes" else None,
        ),
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[0, 2]]
    assert result.outcome_counts.tolist() == [8]
    assert initial.classical.clbits == (1, 2)
    np.testing.assert_array_equal(initial.quantum, [0, 1])


@pytest.mark.parametrize("measure", [False, True])
def test_fast_counts_preserve_unwritten_initial_digits(measure):
    initial = _InitialEvolutionState(classical=_InitialClassicalState(clbits=(1, 2)))
    context = _context(
        n_clbits=2,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    plan = [MeasurementStep((0,), (0,))] if measure else []

    result = _run(
        NumpySVEngine(),
        plan,
        context,
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[0 if measure else 1, 2]]
    assert result.outcome_counts.tolist() == [4]


@pytest.mark.parametrize("digits", [(-1,), (1.5,), (True,), (2**63,)])
def test_initial_register_rejects_invalid_digits(digits):
    with pytest.raises(BackendValidationError, match="non-negative int64"):
        _InitialClassicalState(clbits=digits)


@pytest.mark.parametrize("digits", [(), (1, 2)])
def test_initial_register_requires_matching_width(digits):
    initial = _InitialEvolutionState(classical=_InitialClassicalState(clbits=digits))
    with pytest.raises(BackendValidationError, match="expected 1"):
        initial.classical.validate(
            capabilities=NumpySVEngine().capabilities, n_clbits=1
        )


@pytest.mark.parametrize(
    ("classical", "message"),
    [
        (_InitialClassicalState(clbits=()), "does not support"),
        (_InitialClassicalState(clbits=(0,)), "does not support"),
        (
            _InitialClassicalState(occupied=frozenset()),
            "cannot track carrier occupancy",
        ),
        (
            _InitialClassicalState(occupied=frozenset({0})),
            "cannot track carrier occupancy",
        ),
    ],
)
def test_initial_classical_components_require_engine_support(classical, message):
    initial = _InitialEvolutionState(classical=classical)
    with pytest.raises(BackendValidationError, match=message):
        initial.classical.validate(
            capabilities=NumpyUnitaryEngine().capabilities, n_clbits=1
        )


def test_operator_engine_accepts_absent_classical_components():
    engine = NumpyUnitaryEngine()
    engine.initialize((2,), initial_state=_InitialEvolutionState())
    np.testing.assert_array_equal(engine.export_state(), np.eye(2))


def test_numba_initial_occupancy_preserves_erased_measurement():
    from fatqat.simulator._engine.engines.numba import NumbaSVEngine

    engine = NumbaSVEngine()
    plan = (ResetStep((0,)), MeasurementStep((0,), (0,)))
    initial = _InitialEvolutionState(
        classical=_InitialClassicalState(occupied=frozenset())
    )
    context = _context(
        n_clbits=1,
        shots=4,
        request=_ResultRequest(counts=True, final_state=False),
    )
    result = engine.execute(
        plan,
        context=context,
        simulation_config=_SimulationConfig(
            shot_parallelism="serial", kernel_parallelism="serial"
        ),
        initial_state=initial,
    )

    assert result.outcome_keys.tolist() == [[2]]
    assert result.outcome_counts.tolist() == [4]
