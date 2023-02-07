import logging
from logging import Logger
from typing import Callable, Any
from abc import ABC
import torch

import Analyzer.utils
from .exceptions import AnalyzerNotLoadedError
from .type import Region, Number
from .load import load_breast_biopsy_model
from .result import BreastBiopsyAnalyzerResult
from .analyze import analyze
from Analyzer.utils import get_level

import os
logger = logging.getLogger(__name__)

class AbstractAnalyzer(ABC):
    def __init__(self):
        pass

    @staticmethod
    def is_analyzable(slide_path: str, region) -> bool:
        """
        Check if specified slide_path and region can be analyzed

        Args:
            slide_path: path to slide.
            region: region to run inference.
        """

    def analyze(self, slide_path: str, region) -> Any:
        """
        Infer region of given slide using GPU.

        Args:
            slide_path: path to slide.
            region: region to run inference. ex: ((0,0), (20000, 20000))

        Raises:
            AnalyzerUnsupportedError: specified (slide_path, region) is not supported
        """
        raise NotImplementedError


class BreastBiopsyAnalyzer(AbstractAnalyzer):
    def __init__(self):
        pass

    def load(self):
        """ load breast biopsy model into GPU memory.
                    temporary DataParallel is discontinued
        """
        DEFAULT_CHECKPOINT_PATH = None
        self.model = load_breast_biopsy_model(DEFAULT_CHECKPOINT_PATH)

    def unload(self):
        """ unload breast biopsy model into GPU memory.
                            temporary DataParallel is discontinued
                """
        model = self.model
        self.model = None
        del model
        torch.cuda.empty_cache()
        pass

    @staticmethod
    def is_analyzable(slide_path: str, region: Region, logger: Logger = None) -> bool:
        return True

    def analyze(self, slide_path: str, region: Region, logger: Logger = None, on_progress: Callable[[Number], Any] = (lambda p: None)): #lambda p: None의 의미를 잘 모르겠습니다.
        # if self.model is None:
        #     raise AnalyzerNotLoadedError
        tissue_mask, heatmap_dicts = analyze(self.model, slide_path, region, on_progress=on_progress)
        return BreastBiopsyAnalyzerResult(slide_path, region, heatmap_dicts)


