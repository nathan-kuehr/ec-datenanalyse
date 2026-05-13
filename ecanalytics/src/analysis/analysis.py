
import numpy as np

from copy import deepcopy


from ..data.experiment import Experiment
from ..config import DATA_QUALITY_SERIES_INFO

from .drt import DRT
from .kkt import KKT
from .fitting import Fit
from .regions import Regions


class Analysis:
    Series_Info = DATA_QUALITY_SERIES_INFO

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self.kkt = KKT(root)
        self.drt = DRT(root)
        self.fit = Fit(root)
        self.regions = Regions(root)