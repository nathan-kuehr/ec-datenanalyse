from pyimpspec import Element, Circuit, Frequencies, ComplexImpedances

from abc import abstractmethod

from .model import _set_circuit_parameters

class ReparametrizedElement(Element):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wrapped: Circuit = self._build_wrapped()

    @classmethod
    @abstractmethod
    def _build_wrapped(cls) -> Circuit:
        pass
    
    @classmethod
    @abstractmethod
    def _to_wrapped(cls, **kwargs) -> dict[str, float]:
        pass

    def _impedance(
        self, f: Frequencies, **kwargs,
    ) -> ComplexImpedances:
        _set_circuit_parameters(self._wrapped, self._to_wrapped(**kwargs))
        return self._wrapped.get_impedances(f)