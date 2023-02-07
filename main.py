from Analyzer import BreastBiopsyAnalyzer
import time
from Analyzer.utils import printMem
from openslide import OpenSlide
import os

def print_progress(progress):
    print(f'Progress: {progress}')

printMem("Before loading")
breastAnalyzer = BreastBiopsyAnalyzer()
breastAnalyzer.load()# Creating instance of ProstateAnalyzer loads model to GPU
printMem("After loading")

slide_path = r'/mnt/nfs0/FS91/SMC_Breast_Biopsy/A4-FF-01-0487.tif'
start = time.time()
slide_ = OpenSlide(slide_path)
start_point = [0, 0]
end_point = [slide_.dimensions[0], slide_.dimensions[1]]

#test case
region = {"type":"ellipse", "data": {"min" : start_point, "max":end_point, "angle":30}}

breastAnalyzer.is_analyzable(slide_path, region) #expected pixel values
result = breastAnalyzer.analyze(slide_path, region, on_progress=print_progress)
end = time.time()-start
print(end)
breastAnalyzer.unload()
