"""State owned by one active engine evolution or compiled shot.

System configuration, execution policy, RNGs, and numerical caches live outside
these records. Compiled multi-shot kernels own their state locally per shot.
"""

from dataclasses import dataclass, field
from typing import Generic, NamedTuple, TypeVar

import numpy as np

QuantumDataT = TypeVar("QuantumDataT")


@dataclass(slots=True)
class ClassicalState:
    """Classical container whose components are allocated only when needed.

    Components may stay on the CPU. Updates based on numerical results wait for
    those results and finish before later operations use them.
    """

    # Unallocated register storage leaves unwritten report digits at zero.
    clbits: list[int] | None = None
    # None means implicitly full occupancy; an empty set means no carriers.
    occupied: set[int] | None = None


@dataclass(slots=True)
class EvolutionState(Generic[QuantumDataT]):
    """Native quantum data and classical state owned by one active evolution.

    The engine determines the quantum data type, layout, and representation.
    An empty classical container allocates no component buffers and does not
    indicate capability support. Every evolution receives its own container.
    """

    quantum: QuantumDataT
    classical: ClassicalState = field(default_factory=ClassicalState)


class NumbaClassicalState(NamedTuple):
    """Classical report digits for a compiled shot; occupancy remains implicit."""

    clbits: np.ndarray


class NumbaEvolutionState(NamedTuple):
    """Quantum and classical storage owned by one compiled statevector shot."""

    quantum: np.ndarray
    classical: NumbaClassicalState
