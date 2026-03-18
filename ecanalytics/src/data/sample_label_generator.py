import re


class SampleLabelGenerator:
    """Generates clean sample labels from filename parts using regex rules.

    This class filters out unwanted parts (e.g., standard test names)
    and renames specific patterns to create a concise label for the EIS Experiments.

    Example:
        generator = SampleLabelGenerator(
            drop=[r"EIS", r"PEDOT"],
            rename={r"C([0-9]+)": r"Cleaned_\1"}
        )
        # Input filename: "20251127_EIS_PEDOT_C1_S01.csv"
        label = generator(["20251127", "EIS", "PEDOT", "C1", "S01"])
        # Returns: "20251127_Cleaned_01_S01"
    """

    def __init__(
        self,
        drop: list[str] | None = None,
        rename: dict[str, str] | None = None,
        no_date_drop: bool = False,
    ) -> None:
        """
        Args:
            drop: Regex patterns for parts that should be completely removed.
            rename: Mapping of regex patterns to their replacement strings.
            no_date_drop: If True, the first element (usually the date) is
                          always kept, even if it matches a drop pattern.
        """
        self._drop_patterns = [re.compile(p) for p in (drop or [])]

        self._rename_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in (rename or {}).items()
        ]

        self._no_date_drop = no_date_drop

    def _should_keep(self, index: int, part: str) -> bool:
        """Determines if a filename part should be kept."""
        if index == 0 and self._no_date_drop:
            # Date always at index 0
            return True
        return not any(pat.fullmatch(part) for pat in self._drop_patterns)

    def _apply_renames(self, part: str) -> str:
        """Applies all regex renaming rules sequentially to a part."""
        for pattern, replacement in self._rename_patterns:
            part = pattern.sub(replacement, part)
        return part

    def __call__(self, name_parts: list[str]) -> str:
        """Process the filename parts and generate the final label."""
        if not name_parts:
            return ""

        processed_parts = [
            self._apply_renames(part)
            for index, part in enumerate(name_parts)
            if self._should_keep(index, part)
        ]

        return "_".join(processed_parts)
