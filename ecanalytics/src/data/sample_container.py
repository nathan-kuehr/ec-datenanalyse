import pandas as pd

from collections.abc import Callable

from ..palette import NEIColorPalette


class SampleContainer:
    _container_names_in_use: set[str] = set()
    _CONTAINER_NAME_PREFIX: str | None = None

    def __init__(self, name: str, color: None | str = None) -> None:
        self._palette = NEIColorPalette(color_name=color)

        self._name = ""
        self.name = name

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls._CONTAINER_NAME_PREFIX is None or not isinstance(
            cls._CONTAINER_NAME_PREFIX, str
        ):
            raise ValueError(
                "Sample container subclass needs to implement a name prefix!"
            )

    @classmethod
    def reset_tracked_objects(cls) -> None:
        cls._container_names_in_use = set()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str | None) -> None:
        new_name = new_name or f"{self._CONTAINER_NAME_PREFIX} #{id(self)}"

        if new_name in self._container_names_in_use:
            raise ValueError(
                f"{self._CONTAINER_NAME_PREFIX} name '{new_name}' is already in use. Please choose a unique name."
            )

        self._container_names_in_use.add(new_name)
        self._container_names_in_use.discard(self._name)
        self._name = new_name

    def _apply_container_groups(
        self,
        data: pd.DataFrame,
        name_parts: list[str],
        grouping: dict[str, str | Callable] | None = None,
    ) -> None:
        data[f"{self._CONTAINER_NAME_PREFIX} Name"] = self._name
        data["Palette"] = self._palette

        grouping = grouping or {}

        for group_key, group_value in grouping.items():
            if isinstance(group_value, str):
                data[group_key] = group_value
            elif callable(group_value):
                data[group_key] = group_value(name_parts)
            else:
                raise ValueError(
                    "Grouping values must be either strings or callable functions."
                )
