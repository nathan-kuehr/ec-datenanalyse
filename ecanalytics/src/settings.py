import os
from copy import deepcopy

from .eis import EIS


class Settings:
    def __init__(self, **kwargs):
        self.outputFolder = kwargs.get(
            "outputFolder", GlobalDefaultSettings.outputFolder
        )
        self.exportFormats = kwargs.get(
            "exportFormats", GlobalDefaultSettings.exportFormats
        )

        self.dpi = kwargs.get("dpi", GlobalDefaultSettings.dpi)
        self.showOnSave = kwargs.get("showOnSave", GlobalDefaultSettings.showOnSave)

    @property
    def outputFolder(self) -> str:
        """Folder path for exported plots."""
        return self._outputFolder

    @outputFolder.setter
    def outputFolder(self, folderPath: str) -> None:
        """Sets the global output folder for exports.

        Args:
            folderPath: Path to the output folder
        """
        if not os.path.isdir(folderPath):
            os.makedirs(folderPath, exist_ok=True)

        self._outputFolder = folderPath

    @property
    def exportFormats(self) -> set[str]:
        """Set of file extensions for export formats."""
        return self._exportFormats

    @exportFormats.setter
    def exportFormats(self, formats: set[str]) -> None:
        """Sets the default export formats.

        Args:
            formats: Set of file extensions (e.g., {"svg", "pdf"})
        """
        self._exportFormats = formats & {"svg", "pdf", "png", "jpg"}

    @classmethod
    def cleanKwargs(cls, kwargs: dict, otherKeysToRemove: set[str] = set()) -> dict:
        """Cleans the kwargs dictionary by removing settings related keys.

        Args:
            kwargs: Original kwargs dictionary

        Returns:
            Cleaned kwargs dictionary
        """
        keysToRemove = {
            "outputFolder",
            "exportFormats",
            "dpi",
            "showOnSave",
        } | otherKeysToRemove
        return {k: v for k, v in kwargs.items() if k not in keysToRemove}


# ================================================================================
# Create a global default settings instance
# ================================================================================

StaticGlobalDefaultSettings = Settings.__new__(Settings)

StaticGlobalDefaultSettings._outputFolder = "./vis"
StaticGlobalDefaultSettings._exportFormats = {"svg", "pdf"}
StaticGlobalDefaultSettings.dpi = 300
StaticGlobalDefaultSettings.showOnSave = True

GlobalDefaultSettings = deepcopy(StaticGlobalDefaultSettings)

# ================================================================================


def _set(**kwargs) -> None:
    if "outputFolder" in kwargs:
        GlobalDefaultSettings.outputFolder = kwargs["outputFolder"]

    if "exportFormats" in kwargs:
        GlobalDefaultSettings.exportFormats = kwargs["exportFormats"]

    if "dpi" in kwargs:
        GlobalDefaultSettings.dpi = kwargs["dpi"]

    if "showOnSave" in kwargs:
        GlobalDefaultSettings.showOnSave = kwargs["showOnSave"]


def _reset(args: set) -> None:
    if not isinstance(args, set):
        args = {args}

    if "outputFolder" in args:
        GlobalDefaultSettings.outputFolder = StaticGlobalDefaultSettings.outputFolder

    if "exportFormats" in args:
        GlobalDefaultSettings.exportFormats = StaticGlobalDefaultSettings.exportFormats

    if "dpi" in args:
        GlobalDefaultSettings.dpi = StaticGlobalDefaultSettings.dpi

    if "showOnSave" in args:
        GlobalDefaultSettings.showOnSave = StaticGlobalDefaultSettings.showOnSave

    if EIS in args:
        EIS.resetTrackedObjects()
