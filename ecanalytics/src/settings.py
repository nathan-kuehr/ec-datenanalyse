import os
from copy import deepcopy

from .eis import EIS


class Settings:
    def __init__(self, **kwargs):
        self.output_folder = kwargs.get(
            "outputFolder", Global_Default_Settings.output_folder
        )
        self.export_formats = kwargs.get(
            "exportFormats", Global_Default_Settings.export_formats
        )

        self.dpi = kwargs.get("dpi", Global_Default_Settings.dpi)
        self.show_on_save = kwargs.get(
            "showOnSave", Global_Default_Settings.show_on_save
        )

    @property
    def output_folder(self) -> str:
        """Folder path for exported plots."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, folderPath: str) -> None:
        """Sets the global output folder for exports.

        Args:
            folderPath: Path to the output folder
        """
        if not os.path.isdir(folderPath):
            os.makedirs(folderPath, exist_ok=True)

        self._output_folder = folderPath

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
    def Clean_Kwargs(cls, kwargs: dict, other_keys_to_remove: set[str] = set()) -> dict:
        """Cleans the kwargs dictionary by removing settings related keys.

        Args:
            kwargs: Original kwargs dictionary

        Returns:
            Cleaned kwargs dictionary
        """
        keys_to_remove = {
            "outputFolder",
            "exportFormats",
            "dpi",
            "showOnSave",
        } | other_keys_to_remove
        return {k: v for k, v in kwargs.items() if k not in keys_to_remove}


# ================================================================================
# Create a global default settings instance
# ================================================================================

Static_Global_Default_Settings = Settings.__new__(Settings)

Static_Global_Default_Settings._output_folder = "./vis"
Static_Global_Default_Settings._export_formats = {"svg", "pdf"}
Static_Global_Default_Settings.dpi = 300
Static_Global_Default_Settings.show_on_save = True

Global_Default_Settings = deepcopy(Static_Global_Default_Settings)

# ================================================================================


def _set(**kwargs) -> None:
    if "output_folder" in kwargs:
        Global_Default_Settings.output_folder = kwargs["output_folder"]

    if "export_formats" in kwargs:
        Global_Default_Settings.export_formats = kwargs["export_formats"]

    if "dpi" in kwargs:
        Global_Default_Settings.dpi = kwargs["dpi"]

    if "show_on_save" in kwargs:
        Global_Default_Settings.show_on_save = kwargs["show_on_save"]


def _reset(args: set) -> None:
    if not isinstance(args, set):
        args = {args}

    if "output_folder" in args:
        Global_Default_Settings.output_folder = (
            Static_Global_Default_Settings.output_folder
        )

    if "export_formats" in args:
        Global_Default_Settings.export_formats = (
            Static_Global_Default_Settings.export_formats
        )

    if "dpi" in args:
        Global_Default_Settings.dpi = Static_Global_Default_Settings.dpi

    if "show_on_save" in args:
        Global_Default_Settings.show_on_save = (
            Static_Global_Default_Settings.show_on_save
        )

    if EIS in args:
        EIS.Reset_Tracked_Objects()
