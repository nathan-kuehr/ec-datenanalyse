from __future__ import annotations

import os

from .data.experiment import Experiment


FACTORY_DEFAULTS = {
    "output_folder": os.path.join(".", "vis"),
    "export_formats": {"svg", "pdf"},
    "dpi": 300,
    "show_on_save": True,
}
VALID_EXPORT_FORMATS = {"svg", "pdf", "png", "jpg"}


class Settings:
    Keys = tuple(FACTORY_DEFAULTS)
    def __init__(self, **kwargs) -> None:
        for key in self.Keys:
            setattr(self, key, kwargs.get(key, getattr(_global_defaults, key)))

    @classmethod
    def _from_defaults(cls) -> Settings:
        obj = cls.__new__(cls)
        for key, value in FACTORY_DEFAULTS.items():
            setattr(obj, key, value)
        return obj

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @output_folder.setter
    def output_folder(self, folder_path: str) -> None:
        self._output_folder = folder_path

    @property
    def export_formats(self) -> set[str]:
        return self._export_formats

    @export_formats.setter
    def export_formats(self, formats) -> None:
        self._export_formats = set(formats) & VALID_EXPORT_FORMATS
    
    @classmethod
    def clean_kwargs(cls, kwargs: dict, other_keys_to_remove: set[str] = set()) -> dict:
        remove = set(cls.Keys) | set(other_keys_to_remove)
        return {k: v for k, v in kwargs.items() if k not in remove}


_global_defaults = Settings._from_defaults()


def _set(**kwargs) -> None:
    """Override one or more global default settings."""
    for key in Settings.Keys:
        if key in kwargs:
            setattr(_global_defaults, key, kwargs[key])


def _reset(args: set) -> None:
    """Reset the named settings (and/or tracked Experiments) to their factory defaults."""
    if not isinstance(args, (set, frozenset, list, tuple)):
        args = {args}
    args = set(args)

    for key in Settings.Keys:
        if key in args:
            setattr(_global_defaults, key, FACTORY_DEFAULTS[key])

    if Experiment in args:
        Experiment.reset_tracked_objects()
