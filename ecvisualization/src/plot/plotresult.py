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
        self.__savefigFunc = self.__figure.savefig
        self.__showfigFunc = self.__figure.show
        self.__globalShowFigFunc = plt.show

    def title(self, title: str | None) -> "PlotResult":
        self.__title = title or datetime.now().strftime("%Y-%m-%d_%H:%M:%S_Plot")
        return self

    def show(self) -> "PlotResult":
        self.__showfigFunc()
        self.__globalShowFigFunc(block=True)
        return self

    def save(self) -> "PlotResult":
        for ext in self.__settings.exportFormats:
            exportPath = os.path.join(
                self.__settings.outputFolder, f"{self.__title}.{ext}"
            )

            if ext == "png":
                self.__savefigFunc(exportPath, dpi=self.__settings.dpi)
            else:
                self.__savefigFunc(exportPath)

        self.__saved = True
        return self

    def __condSave(self) -> None:
        if not self.__saved:
            self.save()

    def __enter__(self):
        return self.__figure, self.__figure.axes

    def __exit__(self, excType, excVal, excTb):
        self.__condSave()

    def __del__(self):
        self.__condSave()
        if not self.__settings.showOnSave:
            plt.close(self.__figure)
