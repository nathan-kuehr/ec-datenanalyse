from datetime import datetime
from ..settings import Settings
import matplotlib.pyplot as plt

import os


class PlotResult:
    def __init__(self, title: str | None, figure, **kwargs):
        self.__figure = figure
        self.__title = title or datetime.now().strftime("%Y-%m-%d_%H:%M:%S_Plot")

        self.__saved = kwargs.get("noSave", False)
        self.__settings = Settings(**kwargs)

        # Save References in case of matplotlib shutdown before saving
        self.__savefig_func = self.__figure.savefig
        self.__showfig_func = self.__figure.show
        self.__global_show_fig_func = plt.show

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

    def __enter__(self):
        return self.__figure, self.__figure.axes

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__cond_save()

    def __del__(self):
        self.__cond_save()
        if not self.__settings.show_on_save:
            plt.close(self.__figure)
