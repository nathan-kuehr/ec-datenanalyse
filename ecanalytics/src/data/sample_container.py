import re
import pandas as pd

from ..palette import NEIColorPalette
from collections.abc import Callable
from typing import Any


class SampleContainer:
    __Container_Names_In_Use = set()
    _Container_Name_Prefix: str | None = None

    def __init__(self, name: str, color: None | str = None) -> None:
        self._palette = NEIColorPalette(color_name=color)

        # Naming
        self.__name = ""
        self.name = name

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls._Container_Name_Prefix is None or not isinstance(
            cls._Container_Name_Prefix, str
        ):
            raise ValueError(
                "Sample comtainer subclass needs to implement a name prefix!"
            )

    @classmethod
    def Reset_Tracked_Objects(cls) -> None:
        cls.__Container_Names_In_Use = set()

    @property
    def name(self) -> str:
        return self.__name

    def _container_name_prefix(self):
        pass  # override in subclass

    @name.setter
    def name(self, new_name: str | None) -> None:
        new_name = new_name or f"{self._Container_Name_Prefix} #{id(self)}"

        if new_name in self.__Container_Names_In_Use:
            raise ValueError(
                f"{self._Container_Name_Prefix} name '{new_name}' is already in use. Please choose a unique name."
            )
        else:
            self.__Container_Names_In_Use.add(new_name)
            self.__Container_Names_In_Use.discard(self.__name)
            self.__name = new_name

    def _apply_container_groups(
        self,
        data: pd.DataFrame,
        name_parts: list[str],
        grouping: dict[str, str | Callable] | None = None,
    ):
        data[f"{self._Container_Name_Prefix} Name"] = self.__name
        data["Palette"] = self._palette

        grouping = grouping or {}

        for group_key, g in grouping.items():
            if isinstance(g, str):
                data[group_key] = g
            elif callable(g):
                data[group_key] = g(name_parts)
            else:
                raise ValueError(
                    "Grouping values must be either strings or callable functions."
                )

    class LabelGenerator:
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
            self._drop_patterns = [re.compile(p) for p in (drop or [])]

            self._rename_patterns = [
                (re.compile(pattern), replacement)
                for pattern, replacement in (rename or {}).items()
            ]

            self._no_date_drop = no_date_drop

        def __call__(self, name_parts: list[str]) -> str:
            """Process the filename parts and generate the final label."""
            if not name_parts:
                return ""

            processed_parts = [
                self.__rename(part)
                for i, part in enumerate(name_parts)
                if self.__should_keep(i, part)
            ]

            return "_".join(processed_parts)

        def __should_keep(self, index: int, part: str) -> bool:
            """Determines if a filename part should be kept."""
            if index == 0 and self._no_date_drop:
                # Date always at index 0
                return True
            return not any(pat.fullmatch(part) for pat in self._drop_patterns)

        def __rename(self, part: str) -> str:
            """Applies all regex renaming rules sequentially to a part."""
            for pat, replacement in self._rename_patterns:
                part = pat.sub(replacement, part)
            return part

    class NameExtractor:
        def __init__(self, rule: dict[str, Callable]) -> None:
            self.__rules = {re.compile(k): v for k, v in rule.items()}

        def __call__(self, name_parts: list[str]) -> list[Any] | Any | None:
            results = [self.__extract(part) for part in name_parts]
            results = [r for r in results if r is not None]

            if len(results) > 1:
                return results
            elif len(results) == 1:
                return results[0]
            else:
                return None

        def __extract(self, part: str) -> Any | None:
            for pat, f in self.__rules.items():
                match = pat.search(part)
                if match:
                    return f(match)

            return None
