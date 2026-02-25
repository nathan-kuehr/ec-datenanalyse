import os
import re

import numpy as np


class Import:
    Allowed_File_Extensions = {".txt", ".csv"}

    @classmethod
    def Is_Allowed_File(cls, file_path: str) -> bool:
        """
        Checks if the file has an allowed extension.
        """
        return (
            os.path.isfile(file_path)
            and os.path.splitext(file_path)[1] in cls.Allowed_File_Extensions
        )

    @classmethod
    def Parse_File_Name(cls, file_path: str) -> tuple[int, list[str]] | None:
        """
        Parses the file name to check if it contains a date (YYYYMMDD) and sample number (S##).

        Returns a tuple if ok:
        - int: Sample number
        - list[str]: Parts of the file name
        """
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        name_parts = file_name.split("_")

        date_part = name_parts[0]
        if len(date_part) != 8 or not date_part.isdigit():
            return None

        for i, part in enumerate(name_parts[1:], start=1):
            if part.startswith("S") and part[1:].isdigit():
                sample_number = int(name_parts.pop(i)[1:])
                return (sample_number, name_parts + [f"S{sample_number:02d}"])

        return None

    @classmethod
    def Files_From_Folder(
        cls, folder_path: str, extensions: set[str] | None = None
    ) -> list[str]:
        """Returns a list of file paths from the specified folder, filtering by allowed extensions.

        Args:
            folderPath: Path to folder to search
            extensions: Set of allowed file extensions (default: cls.AllowedFileExtensions)

        Returns:
            List of file paths matching the criteria

        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        if extensions is None:
            extensions = cls.Allowed_File_Extensions

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = []
        for item in os.listdir(folder_path):
            path = os.path.join(folder_path, item)
            if cls.Is_Allowed_File(path):
                files.append(path)

        return files


class FileNameGroupSelector:
    """Smart filename grouping with pattern-based filtering and renaming.

    Example:
        selector = FileNameGroupSelector(
            drop=["EIS", "PEDOT"],
            rename={"C[0-9]+": "Cleaned"}
        )
        group_name = selector(["20251127", "EIS", "PEDOT", "C01"])
        # Returns: "20251127_Cleaned"
    """

    def __init__(
        self,
        drop: list[str] | None = None,
        rename: dict[str, str] | None = None,
        no_date_drop: bool = False,
    ) -> None:
        """Initialize the selector.

        Args:
            drop: List of regex patterns to remove from filename parts
            rename: Dict of regex patterns to replacement strings
            noDateDrop: If False, keep the date part (first element)
        """
        self._drop_pattern = [re.compile(pattern) for pattern in (drop or [])]
        self._rename_pattern = {re.compile(k): v for k, v in (rename or {}).items()}
        self._no_date_drop = no_date_drop

    def _drop(self, part: str) -> bool:
        """Check if a part should be dropped."""
        return any(pat.fullmatch(part) for pat in self._drop_pattern)

    def _rename(self, part: str) -> str:
        """Apply rename patterns to a part."""
        for pattern, repl in self._rename_pattern.items():
            part = pattern.sub(repl, part)
        return part

    def __call__(self, name_parts: list[str]) -> str:
        """Apply filtering and renaming to filename parts.

        Args:
            nameParts: List of filename parts

        Returns:
            Processed group name
        """
        drop_mask = np.array([not self._drop(p) for p in name_parts])
        if self._no_date_drop and len(drop_mask) > 0:
            drop_mask[0] = True

        parts = np.array(name_parts)[drop_mask]
        return "_".join([self._rename(part) for part in parts])
