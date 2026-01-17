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

    def setTitle(self, title: str | None) -> None:
        self.__title = title or datetime.now().strftime("%Y-%m-%d_%H:%M:%S_Plot")

    def save(self) -> None:
        if not self.__saved:
            self.__saved = True

            if self.__settings.showOnSave:
                self.__showfigFunc()
                self.__globalShowFigFunc(block=True)

            for ext in self.__settings.exportFormats:
                exportPath = os.path.join(
                    self.__settings.outputFolder, f"{self.__title}.{ext}"
                )

                if ext == "png":
                    self.__savefigFunc(exportPath, dpi=self.__settings.dpi)
                else:
                    self.__savefigFunc(exportPath)

    def __enter__(self):
        return self.__figure, self.__figure.axes

    def __exit__(self, excType, excVal, excTb):
        pass

    def __del__(self):
        self.save()
        # plt.close(self.__figure)
