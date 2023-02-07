import os
import sys

from openslide import OpenSlide

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Analyzer import ProstateAnalyzer


if __name__ == "__main__":
    prostateAnalyzer = ProstateAnalyzer()  # Loads prostate model to GPU memory
    path = '/path/to/slide'
    slide = OpenSlide(path)
    slide_width, slide_height = slide.dimensions
    result = prostateAnalyzer.analyze(path, ((0, 0), (slide_width, slide_height)))
