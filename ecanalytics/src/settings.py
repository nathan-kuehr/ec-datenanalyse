import os
from copy import deepcopy

from .data.experiment import Experiment


class Settings:
    def __init__(self, **kwargs) -> None:
        self.output_folder = kwargs.get(
            "output_folder", _global_default_settings.output_folder
        )
        self.export_formats = kwargs.get(
            "export_formats", _global_default_settings.export_formats
        )
        self.dpi = kwargs.get("dpi", _global_default_settings.dpi)
        self.show_on_save = kwargs.get(
            "show_on_save", _global_default_settings.show_on_save
        )

    @property
    def output_folder(self) -> str:
        """Folder path for exported plots."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, folder_path: str) -> None:
        """Sets the global output folder for exports.

        Args:
            folder_path: Path to the output folder
        """
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        self._output_folder = folder_path

    @property
    def export_formats(self) -> set[str]:
        """Set of file extensions for export formats."""
        return self._export_formats

    @export_formats.setter
    def export_formats(self, formats: set[str]) -> None:
        """Sets the default export formats.

        Args:
            formats: Set of file extensions (e.g., {"svg", "pdf"})
        """
        self._export_formats = formats & {"svg", "pdf", "png", "jpg"}

    @classmethod
    def clean_kwargs(cls, kwargs: dict, other_keys_to_remove: set[str] = set()) -> dict:
        """Cleans the kwargs dictionary by removing settings related keys.

        Args:
            kwargs: Original kwargs dictionary

        Returns:
            Cleaned kwargs dictionary
        """
        keys_to_remove = {
            "output_folder",
            "export_formats",
            "dpi",
            "show_on_save",
        } | other_keys_to_remove
        return {k: v for k, v in kwargs.items() if k not in keys_to_remove}


# Create a global default settings instance
_static_global_default_settings = Settings.__new__(Settings)

_static_global_default_settings._output_folder = "./vis"
_static_global_default_settings._export_formats = {"svg", "pdf"}
_static_global_default_settings.dpi = 300
_static_global_default_settings.show_on_save = True

_global_default_settings = deepcopy(_static_global_default_settings)


def _set(**kwargs) -> None:
    if "output_folder" in kwargs:
        _global_default_settings.output_folder = kwargs["output_folder"]

    if "export_formats" in kwargs:
        _global_default_settings.export_formats = kwargs["export_formats"]

    if "dpi" in kwargs:
        _global_default_settings.dpi = kwargs["dpi"]

    if "show_on_save" in kwargs:
        _global_default_settings.show_on_save = kwargs["show_on_save"]


def _reset(args: set) -> None:
    if not isinstance(args, set):
        args = {args}

    if "output_folder" in args:
        _global_default_settings.output_folder = (
            _static_global_default_settings.output_folder
        )

    if "export_formats" in args:
        _global_default_settings.export_formats = (
            _static_global_default_settings.export_formats
        )

    if "dpi" in args:
        _global_default_settings.dpi = _static_global_default_settings.dpi

    if "show_on_save" in args:
        _global_default_settings.show_on_save = (
            _static_global_default_settings.show_on_save
        )

    if Experiment in args:
        Experiment.reset_tracked_objects()
