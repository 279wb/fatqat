"""Compiled statevector trajectory sequencing with direct Numba numerical calls.

Each shot owns its quantum and classical state. The workflow interprets the
packed plan, applies conditions, records measurement reports, and advances the
random-draw cursor; numerical modules manipulate the quantum buffer.
"""

import numpy as np
from numba import njit, prange

from .....noise.nb import _report_digit_kernel
from ...numerics.numba._apply import _apply_resolved_range
from ...numerics.numba._channel import _apply_channel_step
from ...numerics.numba._measurement import (
    _sample_index_kernel,
    _sv_probabilities_kernel,
    _sv_project_kernel,
    _sv_reset_kernel,
)
from ...state import NumbaClassicalState, NumbaEvolutionState, NumbaQuantumState


@njit(cache=True)
def _condition_matches(
    clbits, cond_clbit, cond_value, start, length
) -> bool:  # pragma: no cover - compiled by Numba
    """Whether the lowered feedforward condition ``clbits[c] == v`` all hold."""
    for t in range(length):
        if clbits[cond_clbit[start + t]] != cond_value[start + t]:
            return False
    return True


@njit(cache=True)
def _apply_matrix_step(
    state,
    a,
    ap_mat_ptr,
    ap_dim,
    ap_off_ptr,
    ap_comp_ptr,
    ap_comp_len,
    ap_code,
    mat_flat,
    off_flat,
    col_flat,
    val_flat,
    comp_stride_flat,
    comp_dim_flat,
    size,
) -> None:  # pragma: no cover - compiled by Numba
    """Apply compiled gate ``a`` to ``state`` in place (serial coset walk).

    Structure (``ap_code``/columns/values) was resolved when the plan was
    compiled - by declared kernel key or one content scan - so no per-shot
    classification happens here.
    """
    d = ap_dim[a]
    mat_start = ap_mat_ptr[a]
    matrix = mat_flat[mat_start : mat_start + d * d].reshape(d, d)
    off_start = ap_off_ptr[a]
    offsets = off_flat[off_start : off_start + d]
    comp_start = ap_comp_ptr[a]
    comp_end = comp_start + ap_comp_len[a]
    columns = col_flat[off_start : off_start + d]
    values = val_flat[off_start : off_start + d]
    code = ap_code[a]
    _apply_resolved_range(
        state,
        code,
        matrix,
        columns,
        values,
        offsets,
        comp_stride_flat[comp_start:comp_end],
        comp_dim_flat[comp_start:comp_end],
        0,
        size // d,
    )


@njit(cache=True)
def _measure_step(
    state,
    clbits,
    m,
    me_ptr,
    me_len,
    me_classical,
    me_stride,
    me_dim,
    me_conf_ptr,
    conf_flat,
    uniforms,
    draw,
):  # pragma: no cover - compiled by Numba
    """Collapse the measured subsystems, write their reported digits to ``clbits``.

    Takes the shot's uniform pool and its draw cursor rather than a single
    uniform, because the number of draws depends on the noise attached to this
    measurement: one for the Born-sampled collapse, then one more for each
    measured subsystem carrying a readout confusion. The collapse always keeps
    the *true* outcome and only the reported classical value is resampled
    (`_report_digit_kernel`), so state export and qubit reuse are untouched
    while feedforward reads the report - the same semantics, and the same draw
    order, as ``_run_one_shot``'s ``_report_digit`` on the NumPy path.

    Returns the projected state and the advanced cursor.
    """
    start = me_ptr[m]
    length = me_len[m]
    index = _sample_index_kernel(_sv_probabilities_kernel(state), uniforms[draw])
    draw += 1
    state = _sv_project_kernel(
        state, me_stride[start : start + length], me_dim[start : start + length], index
    )
    for j in range(length):
        digit = (index // me_stride[start + j]) % me_dim[start + j]
        conf_ptr = me_conf_ptr[start + j]
        if conf_ptr >= 0:
            digit = _report_digit_kernel(
                conf_flat, conf_ptr, me_dim[start + j], digit, uniforms[draw]
            )
            draw += 1
        clbits[me_classical[start + j]] = digit
    return state, draw


@njit(cache=True)
def _reset_step(
    state, r, rs_ptr, rs_len, rs_stride, rs_dim, u
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Collapse the reset subsystems, then shift each measured branch to |0>."""
    start = rs_ptr[r]
    length = rs_len[r]
    return _sv_reset_kernel(
        state, rs_stride[start : start + length], rs_dim[start : start + length], u
    )


@njit(cache=True, inline="always")
def _initialize_shot_state(start, state_size, custom_start, n_clbits, clbits_start):
    """Allocate one shot's quantum and classical buffers without sharing them."""
    if custom_start:
        # A supplied template must survive unchanged to seed every trajectory.
        quantum = start.copy()
    else:
        # Construct |0...0> directly, without an O(D) template to copy.
        quantum = np.zeros(state_size, dtype=np.complex128)
        quantum[0] = 1.0 + 0.0j
    clbits = (
        clbits_start.copy() if clbits_start.size else np.zeros(n_clbits, dtype=np.int64)
    )
    return NumbaEvolutionState(NumbaQuantumState(quantum), NumbaClassicalState(clbits))


@njit(cache=True, parallel=True)
def _run_sv_trajectories(
    # per-step sequencer (one entry per plan step, in program order)
    step_kind,  # 0=gate, 1=measurement, 2=reset
    step_data,  # row index into this kind's table (gate/measure/reset)
    step_cond_ptr,  # start of this step's condition in the condition pool
    step_cond_len,  # number of (clbit, value) terms in the condition
    # condition pool (flat, shared by all conditioned steps)
    cond_clbit,  # clbit each feedforward term reads
    cond_value,  # value that clbit must equal for the step to fire
    # gate table (one entry per ApplyMatrixStep)
    ap_mat_ptr,  # start of the d*d matrix in mat_flat
    ap_dim,  # local dimension d (matrix is d*d, with d offsets)
    ap_off_ptr,  # start of the d local->flat offsets in off_flat
    ap_comp_ptr,  # start of the complement strides/dims in comp_*_flat
    ap_comp_len,  # number of complement (non-target) subsystems
    ap_code,  # structure code per gate, resolved at compile time
    # gate flat backing
    mat_flat,  # concatenated row-major gate matrices (complex128)
    off_flat,  # concatenated local-index -> flat-offset tables
    col_flat,  # per-row nonzero columns (aligned with off_flat; 0-pad if dense)
    val_flat,  # per-row nonzero values (aligned with off_flat; 0-pad if dense)
    comp_stride_flat,  # concatenated complement strides
    comp_dim_flat,  # concatenated complement dimensions
    # measurement table (one entry per MeasurementStep)
    me_ptr,  # start of this measurement's subsystems in me_*
    me_len,  # number of measured subsystems
    # measurement flat backing
    me_classical,  # clbit each measured subsystem's digit is written to
    me_stride,  # flat stride of each measured subsystem
    me_dim,  # dimension of each measured subsystem (also each confusion's side)
    me_conf_ptr,  # start of this subsystem's confusion in conf_flat, -1 if none
    conf_flat,  # concatenated row-major confusion matrices (float64)
    # reset table (one entry per ResetStep)
    rs_ptr,  # start of this reset's subsystems in rs_*
    rs_len,  # number of reset subsystems
    # reset flat backing
    rs_stride,  # flat stride of each reset subsystem
    rs_dim,  # dimension of each reset subsystem
    # channel table (one entry per ApplyChannelStep), built by `noise.nb`
    ch_kra_ptr,  # start of this channel's Kraus stack in ch_kra_flat
    ch_num_kraus,  # number of Kraus operators in the stack
    ch_dim,  # local dimension d (each operator is d*d, with d offsets)
    ch_off_ptr,  # start of the d local->flat offsets in ch_off_flat
    ch_comp_ptr,  # start of the complement strides/dims in ch_comp_*_flat
    ch_comp_len,  # number of complement (non-target) subsystems
    ch_cdf_ptr,  # start of the fixed branch cdf, -1 if state-dependent
    ch_mmat_diag,  # 1 where every K^dagger K is diagonal: weigh from the marginal
    # channel flat backing (its own pools: a channel carries a stack, not one matrix)
    ch_kra_flat,  # concatenated row-major Kraus operators (complex128)
    ch_mmat_flat,  # concatenated K_i^dagger K_i per Kraus (complex128), for branch weights
    ch_off_flat,  # concatenated local-index -> flat-offset tables
    ch_comp_stride_flat,  # concatenated complement strides
    ch_comp_dim_flat,  # concatenated complement dimensions
    ch_cdf_flat,  # concatenated scaled-unitary branch cdfs, normalized (float64)
    ch_ident_flat,  # 1 where that branch is the identity, so the draw is a no-op
    start,  # custom-state template; empty when the default state is requested
    state_size,  # flat state width, kept explicit when `start` is empty
    custom_start,  # whether each shot must copy `start` rather than build |0...0>
    clbits_start,  # initial report digits; empty for zero initialization
    n_clbits,  # classical-register width: per-shot clbits and result columns
    shots,  # number of independent trajectories - the `prange` extent
    uniforms,  # pre-drawn uniforms, shots*max_draws in execution order
    max_draws,  # per-shot uniform budget; shot s reads uniforms[s*max_draws:]
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Run ``shots`` independent dynamic trajectories in parallel.

    Each shot (a `prange` iteration) owns a local `NumbaEvolutionState`
    and interprets the compiled plan: conditioned gate application, projective
    measurement with readout confusion, conditioned reset, and conditioned channel
    noise. Uniforms are pre-drawn per shot in execution order (slice
    ``uniforms[s*max_draws:]``), consumed one per measurement, one more per
    confusion-bearing measured subsystem, and one per firing reset or channel -
    matching the serial path's RNG stream, so counts are identical. Shots are
    independent, so the result is deterministic regardless of thread scheduling.
    """
    num_steps = step_kind.shape[0]
    results = np.zeros((shots, n_clbits), dtype=np.int64)
    for shot in prange(shots):  # pylint: disable=not-an-iterable
        evolution = _initialize_shot_state(
            start, state_size, custom_start, n_clbits, clbits_start
        )
        draw = shot * max_draws

        for st in range(num_steps):
            kind = step_kind[st]
            passes = _condition_matches(
                evolution.classical.clbits,
                cond_clbit,
                cond_value,
                step_cond_ptr[st],
                step_cond_len[st],
            )
            if kind == 1:  # measurement (unconditional)
                quantum, draw = _measure_step(
                    evolution.quantum.data,
                    evolution.classical.clbits,
                    step_data[st],
                    me_ptr,
                    me_len,
                    me_classical,
                    me_stride,
                    me_dim,
                    me_conf_ptr,
                    conf_flat,
                    uniforms,
                    draw,
                )
                evolution = NumbaEvolutionState(
                    NumbaQuantumState(quantum), evolution.classical
                )
            elif kind == 0 and passes:  # gate
                _apply_matrix_step(
                    evolution.quantum.data,
                    step_data[st],
                    ap_mat_ptr,
                    ap_dim,
                    ap_off_ptr,
                    ap_comp_ptr,
                    ap_comp_len,
                    ap_code,
                    mat_flat,
                    off_flat,
                    col_flat,
                    val_flat,
                    comp_stride_flat,
                    comp_dim_flat,
                    state_size,
                )
            elif kind == 2 and passes:  # reset
                quantum = _reset_step(
                    evolution.quantum.data,
                    step_data[st],
                    rs_ptr,
                    rs_len,
                    rs_stride,
                    rs_dim,
                    uniforms[draw],
                )
                evolution = NumbaEvolutionState(
                    NumbaQuantumState(quantum), evolution.classical
                )
                draw += 1
            elif kind == 3 and passes:  # channel noise (applied in place)
                _apply_channel_step(
                    evolution.quantum.data,
                    step_data[st],
                    ch_kra_ptr,
                    ch_num_kraus,
                    ch_dim,
                    ch_off_ptr,
                    ch_comp_ptr,
                    ch_comp_len,
                    ch_cdf_ptr,
                    ch_mmat_diag,
                    ch_kra_flat,
                    ch_mmat_flat,
                    ch_off_flat,
                    ch_comp_stride_flat,
                    ch_comp_dim_flat,
                    ch_cdf_flat,
                    ch_ident_flat,
                    state_size,
                    uniforms[draw],
                )
                draw += 1

        for c in range(n_clbits):
            results[shot, c] = evolution.classical.clbits[c]
    return results
