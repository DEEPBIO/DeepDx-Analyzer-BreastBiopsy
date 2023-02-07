import os
import sys
import csv
import json
import logging

import cv2
import numpy as np
from openslide import OpenSlide

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Analyzer import BreastResectionAnalyzer

logging.basicConfig(level=logging.DEBUG)

breast_resection_analyzer = BreastResectionAnalyzer()
breast_resection_analyzer.load()


def update_snapshots():
    with open('tests/data/test_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        files_temp = [rows[1] for rows in csv_reader]
        test_files = files_temp[1:]
    for test_file in test_files:
        file_name = test_file.split('/')[-1]

        print(f'Running {test_file}')

        slide = OpenSlide(test_file)

        slide_w, slide_h = slide.dimensions

        region = {
            "type": "rectangle",
            "data": {
                "min": [0, 0],
                "max": [slide_w, slide_h],
                "angle": 0
            }
        }
        result = breast_resection_analyzer.analyze(test_file, region)
        
        cts = {}
        for class_name, heatmap in result.heatmap_dicts.items():
            cts[class_name] = {"contours": None, "hierarchy": None}
            contours, hierarchy = cv2.findContours(heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            contours_list = [contour.tolist() for contour in contours]
            hierarchy_list = hierarchy.tolist()
            cts[class_name]["contours"] = contours_list
            cts[class_name]["hierarchy"] = hierarchy_list

        contour_file_name = ''.join(file_name.split('.')[:-1]) + '.json'
        with open(f'./tests/snapshots/{contour_file_name}', 'w') as outfile:
            json.dump(cts, outfile)


if __name__ == "__main__":
    update_snapshots()
