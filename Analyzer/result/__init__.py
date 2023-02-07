from typing import Tuple, List, Dict, Union, TYPE_CHECKING
import numpy as np
from openslide import OpenSlide
from typing import Tuple, List, Union
from logging import Logger

HeatmapDicts = Dict[str, np.ndarray]

class BreastBiopsyAnalyzerResult():
    def __init__(self, slide_path, region: List[Tuple[int, int]], heatmap_dicts: HeatmapDicts, logger: Logger = None):
        # Note that methods should not change the member variables
        self.input = {
            "slide_path": slide_path,
            "region": region
        }
        self.slide = OpenSlide(slide_path)
        self.heatmap_dicts = heatmap_dicts

    @property
    def tumor_weight(self):
        ratio = [0, 0]
        return ratio
