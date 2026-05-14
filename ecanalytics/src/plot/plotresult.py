from __future__ import annotations

import atexit
import os
import weakref
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..settings import Settings
from . import core


# Weak references to all currently alive but unsaved PlotResults
_alive_unsaved: weakref.WeakSet["PlotResult"] = weakref.WeakSet()


def _flush_unsaved_on_exit() -> None:
    for result in list(_alive_unsaved):
        try:
            result._cond_save()
        except Exception:
            pass


atexit.register(_flush_unsaved_on_exit)


class PlotResult:
    def __init__(self, title: str | None, figure: Figure, **kwargs) -> None:
        self._figure = figure
        self.title = title

        self._saved = kwargs.get("no_save", False)
        self._settings = Settings(**kwargs)

        self._meta: dict[str, Any] = {}

        # Save references in case of matplotlib shutdown before saving
        self._savefig_func = self._figure.savefig
        self._showfig_func = self._figure.show
        self._global_show_fig_func = plt.show

        if not self._saved:
            _alive_unsaved.add(self)

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, title: str | None) -> "PlotResult":
        self._title = title or datetime.now().strftime("%Y-%m-%d_%H:%M:%S_Plot")
        return self

    def show(self) -> "PlotResult":
        self._showfig_func()
        self._global_show_fig_func(block=True)
        return self

    def save(self) -> "PlotResult":
        for ext in self._settings.export_formats:
            export_path = os.path.join(
                self._settings.output_folder, f"{self._title}.{ext}"
            )

            if ext == "png":
                self._savefig_func(export_path, dpi=self._settings.dpi)
            else:
                self._savefig_func(export_path)

        self._saved = True
        _alive_unsaved.discard(self)
        return self

    def _cond_save(self) -> None:
        if not self._saved:
            self.save()
            if not self._settings.show_on_save:
                plt.close(self._figure)

    def __enter__(self) -> tuple[Figure, Axes] | tuple[Figure, list[Axes]]:
        if len(self._figure.axes) == 1:
            return self._figure, self._figure.axes[0]
        return self._figure, self._figure.axes

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cond_save()

    def add_meta(self, new_metadata: dict[str, Any]) -> PlotResult:
        self._meta |= new_metadata
        return self

    def has_meta(self, key: str) -> bool:
        return key in self._meta

    def get_meta(self, key: str) -> Any | None:
        return self._meta.get(key)

    @staticmethod
    def clean_kwargs(kwargs: dict) -> dict:
        return core._clean_args(kwargs, ["no_save"])
