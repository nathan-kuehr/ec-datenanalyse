from __future__ import annotations

import pyimpspec

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from copy import deepcopy
from pyimpspec import Circuit
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
        if label is None or label == "":
            raise ValueError("Each circuit element must be unique and posess an index!")
        
        for symbol in element.get_values().keys():
            yield CircuitElementReference(symbol, f"{symbol}_{label}", element)


def _apply_worker_instruction(
    circuit: Circuit, 
    data: pyimpspec.DataSet, 
    stage: FittingStage,
    hard_lims: dict[str, tuple[float, float]],
) -> None:
    for symbol, label, element in _iterate_elements(circuit):
        value = element.get_value(symbol)

        if label in stage.fix:
            element.set_fixed(symbol, True)
        elif label in stage.vary:
            margin = (stage.vary[label] or np.inf) / 100

            vmin, vmax = value * (1 - margin), value * (1 + margin)
            hmin, hmax = hard_lims[label]

            element.set_fixed(symbol, False) \
                .set_lower_limits(symbol, max(vmin, hmin)) \
                .set_upper_limits(symbol, min(vmax, hmax))
        else:
            element.set_fixed(symbol, False)
        
        # Apply masking
        if not isinstance(stage.mask, np.ndarray):
            raise ValueError("Here, only numpy masks are allowed!")
        data.set_mask({i: ~v for i, v in enumerate(stage.mask)})



class Model(ABC):
    def __init__(self):
        self._lmfit_params = dict()

    def _set_lmfit_params(self, circuit: Circuit) -> Circuit:
        identifiers = pyimpspec.generate_fit_identifiers(circuit)
        for ref in _iterate_elements(circuit):
            self._lmfit_params[ref.label] = identifiers[ref.element][ref.symbol]
        
        return circuit

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
            for symbol, label, element in _iterate_elements(clone):
                element.set_values(symbol, init[label])
            circuits.append(clone)
        return circuits
    
    def _derived_parameters(self, parameters: pd.DataFrame):
        return parameters
