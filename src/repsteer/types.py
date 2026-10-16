from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Pooling = Literal["last", "mean", "hs_mean", "min", "max"]
PositionMode = Literal["all", "last", "generated"]


@dataclass(frozen=True)
class VariableSpec:
    """A binary variable and the nuisance variables held fixed for its vector."""

    name: str
    negative: str
    positive: str
    condition_on: tuple[str, ...] = ()

    def sign_for_target(self, target: str) -> float:
        if target == self.positive:
            return 1.0
        if target == self.negative:
            return -1.0
        raise ValueError(f"Unknown target {target!r} for variable {self.name!r}")

    def opposite(self, value: str) -> str:
        if value == self.negative:
            return self.positive
        if value == self.positive:
            return self.negative
        raise ValueError(f"Unknown value {value!r} for variable {self.name!r}")


@dataclass(frozen=True)
class Intervention:
    layer: int
    vector: np.ndarray
    coefficient: float
    positions: PositionMode = "all"
