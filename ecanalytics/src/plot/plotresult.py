from datetime import datetime
from ..settings import Settings
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import os

from . import core


class PlotResult:
    def __init__(self, title: str | None, figure: Figure, **kwargs):
        self.__figure = figure
        self.title = title

        self.__saved = kwargs.get("no_save", False)
        self.__settings = Settings(**kwargs)

        self.__meta = dict()

        # Save References in case of matplotlib shutdown before saving
        self.__savefig_func = self.__figure.savefig
        self.__showfig_func = self.__figure.show
        self.__global_show_fig_func = plt.show

    @property
    def title(self) -> str:
        return self.__title

    @title.setter
    def title(self, title: str | None) -> "PlotResult":
        self.__title = title or datetime.now().strftime("%Y-%m-%d_%H:%M:%S_Plot")
        return self

    def show(self) -> "PlotResult":
        self.__showfig_func()
        self.__global_show_fig_func(block=True)
        return self

    def save(self) -> "PlotResult":
        for ext in self.__settings.export_formats:
            export_path = os.path.join(
                self.__settings.output_folder, f"{self.__title}.{ext}"
            )

            if ext == "png":
                self.__savefig_func(export_path, dpi=self.__settings.dpi)
            else:
                self.__savefig_func(export_path)

        self.__saved = True
        return self

    def __cond_save(self) -> None:
        if not self.__saved:
            self.save()

    def __enter__(self) -> tuple[Figure, Axes] | tuple[Figure, list[Axes]]:
        if len(self.__figure.axes) == 1:
            return self.__figure, self.__figure.axes[0]
        else:
            return self.__figure, self.__figure.axes

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__cond_save()

    def __del__(self):
        self.__cond_save()
        if not self.__settings.show_on_save:
            plt.close(self.__figure)

    def add_meta(self, new_metadata: dict[str, Any]) -> PlotResult:
        self.__meta |= new_metadata
        return self

    def get_meta(self, key: str) -> Any:
        return self.__meta[key]

    @staticmethod
    def Clean_Kwargs(kwargs: dict):
        return core._clean_args(kwargs, ["no_save"])
