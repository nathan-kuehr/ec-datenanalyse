from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FittingStage:

    fix: list[str] = field(default_factory=list)
    vary: dict[str, float | None] = field(default_factory=dict)
    mask: Callable | np.ndarray | None = None
    constraint_expressions: dict[str, str] = field(default_factory=dict)
    constraint_variables: dict[str, dict] = field(default_factory=dict)

    # def config(self) -> dict[str, float]:
    #     cfg = {p: 0.0 for p in self.fix}
    #     cfg |= {p: np.inf if v is None else v for p, v in self.vary.items()}
    #     return cfg

FittingProcedure = list[FittingStage]