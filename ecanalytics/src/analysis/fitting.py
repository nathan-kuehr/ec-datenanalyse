from __future__ import annotations

import pyimpspec

import numpy as np
import pandas as pd

from copy import deepcopy
from pyimpspec import Circuit, Series, Parallel, Resistor, WarburgOpen, ConstantPhaseElement, ZARC
from scipy import stats, signal as sig
from typing import Callable

from .. import parallel
from ._args import call_argument_parser
from .payloads import ImpedancePayload, FitResultPayload
from ..data.experiment import Experiment
from ..config import (
    FITTING_RANDLES_SERIES_INFO,
    FITTING_DEFAULT_INITIAL_VALUES,
    FITTING_TO_UNIT_CONVERSION_MULTIPLIERS,
    FITTING_DRT_POLARIZATION_CORRECTION_FACTOR,
)


_TURNING_POINT_MIN_FREQUENCY = 50  # Hz
_LOW_FREQUENCY_BAND = (0.1, 1.0)  # Hz, for n_diff angle estimation
_WARBURG_N_UPPER_LIMIT = 0.5


@parallel.CACHE.cache
def _cached_fit_circuit(
    payload: ImpedancePayload, cdc: str, steps: list, **kwargs
) -> FitResultPayload:
    circuit = pyimpspec.parse_cdc(cdc)
    data = pyimpspec.DataSet(payload.frequencies, payload.impedances)

    if len(steps) > 0:
        # Get hard limits from initial circuit setup
        hard_lims = {
            label: (e.get_lower_limit(symbol), e.get_upper_limit(symbol))
            for symbol, label, e in _iterate_circuit_elements(circuit)
        }

        # Iterate over each step of the recipe & fit
        for step in steps:
            FitRecipe._apply_worker_instruction(circuit, data, step, hard_lims)
            fit_result = pyimpspec.fit_circuit(
                circuit=circuit,
                data=data,
                **kwargs,
            )
            circuit = fit_result.circuit
    else:
        # Empty recipe given - fit only once
        fit_result = pyimpspec.fit_circuit(
            circuit=circuit,
            data=data,
            **kwargs,
        )
        circuit = fit_result.circuit

    final_params = {
        label: e.get_value(symbol)
        for symbol, label, e in _iterate_circuit_elements(circuit)
    }

    return FitResultPayload(
        fit_result.frequencies,
        fit_result.impedances,
        fit_result.residuals * 100,
        final_params,
        circuit.to_string(5),
        payload.sample_name,
    )


def _iterate_circuit_elements(circuit: Circuit):
    for element in circuit.get_elements(recursive=True):
        label = element.get_label()
        for symbol in element.get_values().keys():
            yield symbol, f"{symbol}_{label}", element


def _find_turning_point(data: pd.DataFrame) -> float:
    data = data[data["Frequency"] > _TURNING_POINT_MIN_FREQUENCY]

    x = data["Resistance"].to_numpy()
    y = data["Neg. Reactance"].to_numpy()

    indices = sig.find_peaks(x - y)[0]
    if len(indices) == 0:
        raise ValueError("No turning point found!")

    # Take last peak as freq. values are ordered high to low
    f = data["Frequency"].iloc[indices[-1]]
    print(f"Turning point: {f} Hz - {data['Sample Name'].unique()}")
    return f


class Fit:
    SERIES_INFO = FITTING_RANDLES_SERIES_INFO
    _DEFAULT_INITIAL_VALUES = FITTING_DEFAULT_INITIAL_VALUES
    _DEFAULT_CALCULATION_ARGS = {
        "method": "least_squares",
        "weight": "boukamp",
        "max_nfev": -1,  # Unlimited
        "num_procs": 1,
    }

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self._data: pd.DataFrame | None = None
        self._params: pd.DataFrame | None = None
        self._circuits: list[Circuit] | None = None

        self._mask_list: list[np.ndarray] = []
        self.set_mask(None)

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
        return self._root._add_metadata_to_data(melted)

    def simulate(self, frequencies: np.ndarray) -> pd.DataFrame:
        if self._circuits is None:
            raise ValueError("No fitting done yet - please call fit() explicitly!")

        omegas = 2 * np.pi * frequencies

        dfs = []
        for sample, circuit in zip(self._root._samples, self._circuits):
            impedances = circuit.get_impedances(frequencies)

            with np.errstate(divide="ignore"):
                capacitances = np.abs(1 / (omegas * impedances.imag))

            offset_corrected = (
                impedances.real
                - sample._resistance_offset
                + self._root.mean_resistance_offset
                + self._root._resistive_shift
            )

            dfs.append(pd.DataFrame({
                "Frequency": frequencies,
                "Impedance": np.abs(impedances),
                "Resistance": impedances.real,
                "Neg. Reactance": -impedances.imag,
                "Phase": np.degrees(np.angle(impedances)),
                "Capacitance": capacitances,
                "Offset-Corrected Resistance": offset_corrected,
                "Sample Name": sample._name,
            }))

        data = pd.concat(dfs, axis=0, ignore_index=True)
        return self._root._add_metadata_to_data(data)

    def simulate_experiment(self, frequencies: np.ndarray) -> Experiment:
        sim = self._root.phantom(self.simulate(frequencies))

        self._root.data["Data Origin"] = "Measured"
        sim.data["Data Origin"] = "Simulated Fitted"

        return sim

    def set_mask(self, mask: Callable | np.ndarray | None = None) -> None:
        self._mask_list = self._make_masks(mask)

    def _make_masks(
        self, mask: Callable | np.ndarray | None = None
    ) -> list[np.ndarray]:
        nsamples = self._root.data["Sample Name"].nunique()
        nfreqs = self._root.data["Frequency"].nunique()

        def check_1d_mask(m: np.ndarray) -> np.ndarray:
            if m.ndim != 1 or len(m) != nfreqs:
                raise ValueError("The given mask has not the correct shape for the underlying data!")
            return m

        if mask is None:
            return [np.ones(nfreqs, dtype=bool)] * nsamples
        elif isinstance(mask, np.ndarray):
            return [check_1d_mask(mask)] * nsamples
        elif isinstance(mask, Callable):
            return [
                check_1d_mask(mask(name, group))
                for name, group in self._root.data.groupby("Sample Name", sort=False)
            ]
        else:
            raise ValueError("Invalid mask type!")

    def __call__(
        self,
        circuit: Circuit | list[Circuit],
        recipe: FitRecipe | None = None,
        *args,
        **kwargs,
    ) -> None:
        data = self._root.data
        sample_names = self._root.sample_names

        nsamples = len(sample_names)

        if isinstance(circuit, Circuit):
            circuit = [circuit] * nsamples
        if not len(circuit) == nsamples:
            raise ValueError("Insufficient list of circuits provided!")
        cdc_list = [c.to_string(3) for c in circuit]

        # Instructions
        instr_list = (recipe or FitRecipe())._to_worker_instructions(self)

        args_list = call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, sample_names, "DRT"
        )
        args_list = [
            al | {"cdc": cdc, "steps": instr}
            for al, cdc, instr in zip(args_list, cdc_list, instr_list)
        ]

        input_list = ImpedancePayload.from_data(data, self._mask_list)

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
                pd.DataFrame([self._convert_fit_to_unit_parameters(payload.parameters)])
                .assign(**{"Sample Name": payload.sample_name})
            )
            circuits.append(pyimpspec.parse_cdc(payload.cdc))
        freq_df = pd.concat(freq_dfs, axis=0, ignore_index=True)
        param_df = pd.concat(param_dfs, axis=0, ignore_index=True)

        self._data = self._root._add_metadata_to_data(freq_df)
        self._params = self._root._add_metadata_to_data(param_df)
        self._circuits = circuits

    @staticmethod
    def make_randles(
        pre_init: tuple[Experiment, float] | None = None,
    ) -> tuple[Circuit | list[Circuit], FitRecipe]:
        r_s = Resistor().set_label("s")
        r_ct = Resistor().set_label("ct")

        q_dl = ConstantPhaseElement().set_label("dl")

        w_diff = (
            WarburgOpen()
            .set_label("diff")
            .set_fixed(Y=False, n=False, B=False)
            .set_upper_limits(n=_WARBURG_N_UPPER_LIMIT)
        )

        randles = Circuit(Series([r_s, Parallel([Series([r_ct, w_diff]), q_dl])]))

        if pre_init is not None:
            experiment, charac_peak_tau = pre_init

            circuit: list[Circuit] | Circuit = []

            inits = Fit._estimate_initial_values_randles(
                data=experiment.data,
                peak_data=experiment.analysis.drt.peak_select([charac_peak_tau]),
                masks=experiment.analysis.fit._mask_list,
            )
            for init in inits:
                for symbol, label, element in _iterate_circuit_elements(randles):
                    element.set_values(symbol, init[label])
                circuit.append(deepcopy(randles))
        else:
            circuit = randles

        return circuit, FitRecipe.randles()

    @staticmethod
    def make_zarc(
        pre_init: tuple[Experiment, float] | None = None,
    ) -> tuple[Circuit | list[Circuit], FitRecipe]:
        rq = ZARC().set_label("zarc")

        original = Circuit(Series([rq]))

        if pre_init is not None:
            experiment, charac_peak_tau = pre_init

            circuit: list[Circuit] | Circuit = []

            inits = Fit._estimate_initial_values_zarc(
                peak_data=experiment.analysis.drt.peak_select([charac_peak_tau])
            )
            for init in inits:
                for symbol, label, element in _iterate_circuit_elements(original):
                    element.set_values(symbol, init[label])
                circuit.append(deepcopy(original))
        else:
            circuit = original

        return circuit, FitRecipe()

    @staticmethod
    def _convert_fit_to_unit_parameters(parameters: dict[str, float]) -> dict[str, float]:
        return {
            k: v * FITTING_TO_UNIT_CONVERSION_MULTIPLIERS.get(k, 1)
            for k, v in parameters.items()
        }

    @classmethod
    def _estimate_initial_values_randles(
        cls,
        data: pd.DataFrame,
        peak_data: pd.DataFrame,
        masks: list[np.ndarray],
    ) -> list[dict]:
        freqs = data["Frequency"].unique()
        low_freq_min, low_freq_max = _LOW_FREQUENCY_BAND
        low_freq_mask = (freqs >= low_freq_min) & (freqs <= low_freq_max)

        nsamples = data["Sample Name"].nunique()

        # Offset / solution resistance -> fill up later
        r_s = np.empty(nsamples)

        # Charge transfer resistance
        r_ct = (
            peak_data["Polarisation"].to_numpy()
            * FITTING_DRT_POLARIZATION_CORRECTION_FACTOR
        )

        # Double layer CPE
        n_dl = np.ones_like(r_ct) * cls._DEFAULT_INITIAL_VALUES["randles"]["n_dl"]

        taus = 10 ** peak_data["Log. Position"].to_numpy()
        y_dl = np.power(taus, n_dl) / r_ct

        # Reflective Warburg element
        b_diff = np.ones_like(r_ct) * cls._DEFAULT_INITIAL_VALUES["randles"]["B_diff"]
        y_diff = np.ones_like(r_ct) * cls._DEFAULT_INITIAL_VALUES["randles"]["Y_diff"]
        n_diff = np.empty_like(r_ct)

        for i, (_, group) in enumerate(data.groupby("Sample Name", sort=False)):
            impedance_data = group[["Resistance", "Neg. Reactance"]].to_numpy()

            # Angle estimation for n_diff
            # Fit ΔR / ΔX for stability as we approach 90° here
            m_inv = stats.linregress(
                x=impedance_data[low_freq_mask, 1],
                y=impedance_data[low_freq_mask, 0],
            ).slope
            n_diff[i] = (np.arctan(1 / m_inv) / np.pi) if m_inv != 0.0 else 0.5

            # Solution resistance estimation
            r_s[i] = np.min(impedance_data[masks[i], 0])

        return pd.DataFrame({
            "R_s": r_s,
            "R_ct": r_ct,
            "Y_dl": y_dl,
            "n_dl": n_dl,
            "B_diff": b_diff,
            "Y_diff": y_diff,
            "n_diff": n_diff,
        }).to_dict(orient="records")

    @classmethod
    def _estimate_initial_values_zarc(cls, peak_data: pd.DataFrame) -> list[dict]:
        r = (
            peak_data["Polarisation"].to_numpy()
            * FITTING_DRT_POLARIZATION_CORRECTION_FACTOR
        )
        tau = 10 ** peak_data["Log. Position"].to_numpy()
        n = np.ones_like(r) * cls._DEFAULT_INITIAL_VALUES["zarc"]["n_zarc"]

        return pd.DataFrame({
            "R_zarc": r,
            "tau_zarc": tau,
            "n_zarc": n,
        }).to_dict(orient="records")


class FitRecipe:
    def __init__(self) -> None:
        self._steps: list[tuple[dict, Callable | np.ndarray | None]] = []

    def add_step(
        self,
        fix: list[str] = [],
        vary: dict[str, float | None] = {},
        mask: Callable | np.ndarray | None = None,
    ) -> FitRecipe:
        config = {p: 0.0 for p in fix}
        config |= {p: np.inf if v is None else v for p, v in vary.items()}

        self._steps.append((config, mask))

        return self

    def _to_worker_instructions(self, fit: Fit) -> list[list]:
        nsamples = len(fit._mask_list)

        instructions: list[list] = [[] for _ in range(nsamples)]

        for config, mask in self._steps:
            for i, m, sm in zip(instructions, fit._mask_list, fit._make_masks(mask)):
                i.append((config, sm[m]))

        return instructions

    @staticmethod
    def randles() -> FitRecipe:
        # Mask generator for the semicircle
        def semicircle_mask_gen(_, x: pd.DataFrame) -> np.ndarray:
            return x["Frequency"].unique() >= _find_turning_point(x)

        return (
            FitRecipe()
            .add_step(
                fix=["B_diff", "Y_diff", "n_diff"],
                mask=semicircle_mask_gen,
            )
            .add_step(
                fix=["R_s"],
                vary={
                    "R_ct": 5,
                    "Y_dl": 5,
                    "n_dl": 5,
                    "B_diff": None,
                    "Y_diff": None,
                    "n_diff": None,
                },
            )
            .add_step(
                vary={
                    "R_s": None,
                    "R_ct": None,
                    "Y_dl": None,
                    "n_dl": None,
                    "B_diff": None,
                    "Y_diff": None,
                    "n_diff": None,
                }
            )
        )

    @staticmethod
    def _apply_worker_instruction(
        circuit: Circuit,
        data: pyimpspec.DataSet,
        step: tuple[dict, np.ndarray],
        hard_lims: dict,
    ) -> None:
        config, mask = step

        for symbol, label, element in _iterate_circuit_elements(circuit):
            value = element.get_value(symbol)

            margin = config.get(label, np.inf) / 100
            vmin, vmax = value * (1 - margin), value * (1 + margin)
            hmin, hmax = hard_lims[label]

            # Fix element if no margin allowed
            element.set_fixed(symbol, margin == 0.0)

            if margin > 0:
                element.set_lower_limits(symbol, max(vmin, hmin))
                element.set_upper_limits(symbol, min(vmax, hmax))

            data.set_mask({i: ~v for i, v in enumerate(mask)})
