from __future__ import annotations

import numpy as np
import pandas as pd
import pyimpspec

from dataclasses import replace
from pyimpspec import Circuit
from typing import Callable

from .model import Model, _iterate_elements, _apply_worker_instruction
from .stage import FittingStage, FittingProcedure
from .._args import call_argument_parser
from ..kkt import as_component_data
from ..payloads import ImpedancePayload, FitResultPayload
from ... import parallel
from ...config import FITTING_RANDLES_SERIES_INFO
from ...data.experiment import Experiment, SimulatedExperiment


@parallel.CACHE.cache
def _cached_fit_circuit(
    payload: ImpedancePayload, cdc: str, procedure: FittingProcedure, **kwargs
) -> FitResultPayload:
    circuit = pyimpspec.parse_cdc(cdc)
    data = pyimpspec.DataSet(payload.frequencies, payload.impedances)

    if len(procedure) > 0:
        hard_lims = {
            label: (e.get_lower_limit(symbol), e.get_upper_limit(symbol))
            for symbol, label, e in _iterate_elements(circuit)
        }

        for stage in procedure:
            _apply_worker_instruction(circuit, data, stage, hard_lims)
            ce = stage.constraint_expressions
            cv = stage.constraint_variables
            
            fit_result = pyimpspec.fit_circuit(
                circuit=circuit,
                data=data,
                constraint_expressions=ce or None,
                constraint_variables=cv or None,
                **kwargs,
            )
            circuit = fit_result.circuit
    else:
        fit_result = pyimpspec.fit_circuit(
            circuit=circuit,
            data=data,
            **kwargs,
        )
        circuit = fit_result.circuit

    final_params = {
        label: e.get_value(symbol)
        for symbol, label, e in _iterate_elements(circuit)
    }

    return FitResultPayload(
        fit_result.frequencies,
        fit_result.impedances,
        fit_result.residuals * 100,
        final_params,
        circuit.to_string(5),
        payload.sample_name,
    )


class Fit:
    SERIES_INFO = FITTING_RANDLES_SERIES_INFO
    _DEFAULT_CALCULATION_ARGS = {
        "method": "least_squares",
        "weight": "boukamp",
        "max_nfev": -1,  # Unlimited
        "num_procs": 1,
    }

    def __init__(self, root: "Experiment") -> None:
        self._root = root

        self._data: pd.DataFrame | None = None
        self._params: pd.DataFrame | None = None
        self._circuits: list[Circuit] | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("No fitting done yet - please call fit() explicitly!")
        return self._data
    
    @property
    def params(self) -> pd.DataFrame:
        if self._params is None:
            raise ValueError("No fitting done yet - please call fit() explicitly!") 
        return self._params

    @property
    def params_long(self) -> pd.DataFrame:
        melted = self.params.melt(
            id_vars=["Sample Name"],
            value_vars=list(self.params.loc[:, :"Sample Name"].columns)[:-1],
            var_name="Parameter",
            value_name="Value",
        )[["Parameter", "Value", "Sample Name"]]
        return self._add_metadata_to_data(melted)

    def __call__(
        self,
        model: Model | Circuit | list[Circuit],
        procedure: FittingProcedure | list[FittingProcedure] = [],
        *args,
        **kwargs,
    ) -> None:
        assert not isinstance(self._root, SimulatedExperiment)

        # Lengths
        sample_names = self._root.sample_names
        nsamples = len(sample_names)

        # Convert to list[Circuit], list[list[FittingStage]]
        if isinstance(model, Model):
            circuit = model._prepare(self._root)
            procedure = model._stages(self._root)
        else:
            circuit = model
        if isinstance(circuit, Circuit):
            circuit = [circuit] * nsamples
        if isinstance(procedure[0], FittingStage):
            procedure = [procedure] * nsamples

        if not len(circuit) == nsamples == len(procedure):
            raise ValueError("Insufficient list of circuits or procedures provided!")
        
        # Prepare CDCs and procedures
        cdc_list = [c.to_string(3) for c in circuit]
        procedure_list = self._materialize_masks(procedure)
        
        # Prepare args and add cdc & procedure
        args_list = call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, sample_names, "Fit"
        )
        args_list = [
            al | {"cdc": cdc, "procedure": proc}
            for al, cdc, proc in zip(args_list, cdc_list, procedure_list)
        ]

        # Input is the data
        input_list = ImpedancePayload.from_data(self._root.data)

        payloads: list[FitResultPayload] = parallel.multiprocess(
            method=_cached_fit_circuit,
            inputs=input_list,
            tqdm_note=f"Fitting circuit to EIS Experiment '{self._root.name}'",
            args=args_list,
        )

        freq_dfs = []
        param_dfs = []
        circuits = []
        for payload in payloads:
            freq_dfs.append(
                pd.DataFrame(
                    {
                        "Frequency": payload.frequencies,
                        "Resistance": np.real(payload.impedances),
                        "Neg. Reactance": -np.imag(payload.impedances),
                        "Real Residual": np.real(payload.residuals),
                        "Imag. Residual": np.imag(payload.residuals),
                        "Sample Name": payload.sample_name,
                    }
                )
            )
            param_dfs.append(
                pd.DataFrame([payload.parameters])
                .assign(**{"Sample Name": payload.sample_name})
            )
            circuits.append(pyimpspec.parse_cdc(payload.cdc))
        freq_df = pd.concat(freq_dfs, axis=0, ignore_index=True)
        param_df = pd.concat(param_dfs, axis=0, ignore_index=True)

        # Add additional parameters that can be derived form the fitted
        if isinstance(model, Model):
            param_df = model._post_process_parameters(param_df)
            
        sorted = [c for c in param_df.columns if c != "Sample Name"] + ["Sample Name"]
        param_df = param_df[sorted]

        self._data = self._add_metadata_to_data(freq_df)
        self._params = self._add_metadata_to_data(param_df)
        self._circuits = circuits

    def simulate(self, frequencies: np.ndarray | None = None) -> pd.DataFrame:
        assert not isinstance(self._root, SimulatedExperiment)
        if self._circuits is None:
            raise ValueError("No fitting done yet - please call fit() explicitly!")

        if frequencies is None:
            frequencies = self.data["Frequency"].unique()

        frequencies = np.sort(frequencies)[::-1]
        omegas = 2 * np.pi * frequencies

        dfs = []
        for sample, circuit in zip(self._root._samples, self._circuits):
            impedances = circuit.get_impedances(frequencies)

            with np.errstate(divide="ignore"):
                capacitances = np.abs(1 / (omegas * impedances.imag))

            oc_impedance = impedances - sample._resistance_offset
            oc_resistance = oc_impedance.real
            oc_phase = np.degrees(np.arctan2(oc_impedance.imag, oc_resistance))

            dfs.append(pd.DataFrame({
                "Frequency": frequencies,
                "Impedance": np.abs(impedances),
                "Resistance": impedances.real,
                "Neg. Reactance": -impedances.imag,
                "Phase": np.degrees(np.angle(impedances)),
                "Capacitance": capacitances,
                "Offset-Corrected Resistance": oc_resistance + self._root.mean_resistance_offset,
                "Offset-Corrected Impedance": np.abs(oc_impedance),
                "Offset-Corrected Phase": oc_phase,
                "Sample Name": sample._name,
            }))

        data = pd.concat(dfs, axis=0, ignore_index=True)
        return self._add_metadata_to_data(data)

    def simulate_experiment(self, frequencies: np.ndarray | None = None) -> SimulatedExperiment:
        assert not isinstance(self._root, SimulatedExperiment)

        sim = self._root.phantom()

        main_data = self.simulate(frequencies)
        kkt_data = self._add_metadata_to_data(as_component_data(self.data))

        sim._data = main_data
        sim.analysis.kkt._data = kkt_data

        return sim
    
    def _add_metadata_to_data(self, data: pd.DataFrame):
        data = self._root._add_metadata_to_data(data)
        data["Data Origin"] = "Fitted"
        return data
    
    def _materialize_masks(self, old_procedures: list[FittingProcedure]) -> list[FittingProcedure]:
        nfreqs = self._root.data["Frequency"].nunique()
        grouped = self._root.data.groupby("Sample Name", sort=False)

        def resolve_mask(mask_attr: Callable | np.ndarray | None, name: str, group: pd.DataFrame) -> np.ndarray:
            if mask_attr is None:
                m = np.ones(nfreqs, dtype=bool)
            elif isinstance(mask_attr, np.ndarray):
                m = mask_attr
            elif callable(mask_attr):
                m = mask_attr(name, group)
            else:
                raise TypeError(f"Invalid mask type: {type(mask_attr)}")
            
            if not isinstance(m, np.ndarray) or m.shape != (nfreqs,):
                raise ValueError("The given mask does not have the correct shape for the underlying data!")
            
            return m
        
        return [
            [
                replace(stage, mask=resolve_mask(stage.mask, name, group))
                for stage in old_procedure
            ]
            for old_procedure, (name, group) in zip(old_procedures, grouped)
        ]
