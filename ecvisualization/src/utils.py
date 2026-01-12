import os
import re

import pandas as pd
import numpy as np

class Import:
    AllowedFileExtensions = {'.txt', '.csv'}

    @classmethod
    def isAllowedFile(cls, filePath: str) -> bool:
        """
        Checks if the file has an allowed extension.
        """
        return os.path.isfile(filePath) and os.path.splitext(filePath)[1] in cls.AllowedFileExtensions

    @classmethod
    def parseFileName(cls, filePath: str) -> tuple[int, list[str]]|None:
        """
        Parses the file name to check if it contains a date (YYYYMMDD) and sample number (S##).

        Returns a tuple if ok:
        - int: Sample number
        - list[str]: Parts of the file name
        """
        fileName = os.path.splitext(os.path.basename(filePath))[0]
        nameParts = fileName.split('_')
        
        datePart = nameParts[0]
        if len(datePart) != 8 or not datePart.isdigit():
            return None
        
        for i, part in enumerate(nameParts[1:], start=1):
            if part.startswith('S') and part[1:].isdigit():
                sampleNumber = int(nameParts.pop(i)[1:])
                return (sampleNumber, nameParts + [f"S{sampleNumber:02d}"])

        return None
    
    @classmethod
    def filesFromFolder(cls, folderPath: str, extensions: set[str] | None = None) -> list[str]:
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
            extensions = cls.AllowedFileExtensions
            
        if not os.path.isdir(folderPath):
            raise FileNotFoundError(f"Folder not found: {folderPath}")
        
        files = []
        for item in os.listdir(folderPath):
            path = os.path.join(folderPath, item)
            if cls.isAllowedFile(path):
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
        noDateDrop: bool = False
    ) -> None:
        """Initialize the selector.
        
        Args:
            drop: List of regex patterns to remove from filename parts
            rename: Dict of regex patterns to replacement strings
            noDateDrop: If False, keep the date part (first element)
        """
        self._dropPattern = [re.compile(pattern) for pattern in (drop or [])]
        self._renamePattern = {re.compile(k): v for k, v in (rename or {}).items()}
        self._noDateDrop = noDateDrop

    def _drop(self, part: str) -> bool:
        """Check if a part should be dropped."""
        return any(pat.fullmatch(part) for pat in self._dropPattern)
    
    def _rename(self, part: str) -> str:
        """Apply rename patterns to a part."""
        for pattern, repl in self._renamePattern.items():
            part = pattern.sub(repl, part)
        return part
    
    def __call__(self, nameParts: list[str]) -> str:
        """Apply filtering and renaming to filename parts.
        
        Args:
            nameParts: List of filename parts
            
        Returns:
            Processed group name
        """
        dropMask = np.array([not self._drop(p) for p in nameParts])
        if self._noDateDrop and len(dropMask) > 0:
            dropMask[0] = True

        parts = np.array(nameParts)[dropMask]
        return "_".join([self._rename(part) for part in parts])
    
class CovarianceVisualization:

    @classmethod
    def calculate(cls, data: pd.DataFrame, real: str = "Offset-Corrected Resistance", imag: str = "Neg. Reactance"):
        nF = data["Frequency"].nunique()
        N = data["Name"].nunique()

        grouped = data.groupby("Frequency")[[real, imag]]

        f = grouped.index.to_numpy()
        covs = (grouped.cov() / N).to_numpy().reshape(nF, 2, 2)

        return np.column_stack((f, covs))
    
    @classmethod
    def ellipseParameters(cls, covData: np.ndarray, ci: float = 0.95) -> np.ndarray:
        from scipy.stats import chi2

        chi2Val = chi2.ppf(ci, df=2)

        ellipses = []
        for freq, cov in covData:
            eigvals, eigvecs = np.linalg.eigh(cov)
            axes = np.sqrt(eigvals * chi2Val)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipses.append((freq, axes[0], axes[1], angle))
        
        return np.array(ellipses)
