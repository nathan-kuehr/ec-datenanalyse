from __future__ import annotations

import pyimpspec

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from copy import deepcopy
from pyimpspec import Circuit, DataSet
from typing import Iterator, NamedTuple

from .stage import FittingStage, FittingProcedure
from ...data.experiment import Experiment


class CircuitElementReference(NamedTuple):
    symbol: str
    label: str
    element: pyimpspec.Element


def _iterate_elements(circuit: Circuit) -> Iterator[CircuitElementReference]:
    for element in circuit.get_elements(recursive=True):
        label = element.get_label()
        
        for symbol in element.get_values().keys():
            yield CircuitElementReference(symbol, f"{symbol}_{label}", element)


def _apply_worker_instruction(
    circuit: Circuit, 
    data: DataSet, 
    stage: FittingStage,
    hard_lims: dict[str, tuple[float, float]],
) -> None:
    for symbol, label, element in _iterate_elements(circuit):
        key = label.strip("_")
        value = element.get_value(symbol)

        if key in stage.fix:
            element.set_fixed(symbol, True)
        elif key in stage.vary:
            # vary holds a +/- percent band around the current parameter value
            # (in log space for the log_ parameters); None means unbounded.
            # min/max keeps lo <= hi even when the value is negative.
            frac = np.inf if stage.vary[key] is None else stage.vary[key] / 100

            b1, b2 = value * (1 - frac), value * (1 + frac)
            lo, hi = min(b1, b2), max(b1, b2)

            hmin, hmax = hard_lims[label]

            element.set_fixed(symbol, False) \
                .set_lower_limits(symbol, max(lo, hmin)) \
                .set_upper_limits(symbol, min(hi, hmax))
        else:
            element.set_fixed(symbol, False)
        
        # Apply masking
        if not isinstance(stage.mask, np.ndarray):
            raise ValueError("Here, only numpy masks are allowed!")
        data.set_mask({i: ~v for i, v in enumerate(stage.mask)})


def _set_circuit_parameters(circuit: Circuit, vals: dict[str, float]):
    for symbol, label, element in _iterate_elements(circuit):
        if (v := vals.get(label.strip("_"))) is not None:
            element.set_values(symbol, v)



class ReparametrizationMixin(ABC):
    def __init__(self) -> None:
        self._wrapped = self._build_wrapped()

    @abstractmethod
    def _build_wrapped(self) -> Circuit:
        pass
    
    @abstractmethod
    def _to_wrapped(self, **kwargs):
        pass

    
    

class Model(ABC):
    @abstractmethod
    def _build(self) -> Circuit:
        pass

    def _initial_values(self, exp: Experiment) -> list[dict] | None:
        return None

    def _stages(self, exp: Experiment) -> list[FittingProcedure]:
        return [[] for _ in exp.sample_names] # Single shot fitting

    def _prepare(self, exp: Experiment) -> list[Circuit]:
        base = self._build()

        # Check if we can initialize
        if (inits := self._initial_values(exp)) is None:
            return [ deepcopy(base) for _ in exp.sample_names]

        circuits: list[Circuit] = []
        for init in inits:
            clone = deepcopy(base)
            _set_circuit_parameters(clone, init)
            circuits.append(clone)
        return circuits
    
    def _post_process_parameters(self, parameters: pd.DataFrame):
        return parameters
