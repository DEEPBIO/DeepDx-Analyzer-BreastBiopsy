import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import csv
import json
import unittest
from unittest import mock
import logging

import torch
from openslide import OpenSlide
import numpy as np
import cv2

from Analyzer import BreastResectionAnalyzer

logging.basicConfig(level=logging.DEBUG)

breast_resection_analyzer = BreastResectionAnalyzer()
breast_resection_analyzer.load()
start_memory = torch.cuda.memory_allocated()

with open('tests/data/test_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    files_temp = [rows[1] for rows in csv_reader]
    test_files = files_temp[1:]

if not os.path.exists('./tests/__test_run__/'):
    os.makedirs('./tests/__test_run__/')


class E2eInvarianceTest(unittest.TestCase):
    def tearDownClass() -> None:
        print(f"************************************")

        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(gpu_idx) for gpu_idx in range(num_gpus)]
        print(f"GPUs: {gpu_names}")

    def test_omitting_optional_args(self):
        try:
            slide_path = "tests/data/CMU-1-Small-Region.svs"
            slide = OpenSlide(slide_path)
            slide_w, slide_h = slide.dimensions
            region = {
                "type": "rectangle",
                "data": {
                    "min": [0, 0],
                    "max": [slide_w, slide_h],
                    "angle": 0
                }
            }
            breast_resection_analyzer.analyze("tests/data/CMU-1-Small-Region.svs", region)
        except TypeError:
            self.fail()
        # pylint: disable-next=broad-except
        except Exception:
            # for errors that are not TypeError -> success
            pass

    def test_result_is_equal_to_snapshot(self):
        elapsed_time = 0
        for test_file in test_files:
            with self.subTest(msg=test_file):

                print(f'Testing file {test_file}')

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
                mock_function = mock.Mock()
                start_time = time.time()
                result = breast_resection_analyzer.analyze(test_file, region, on_progress=mock_function)
                analysis_time = time.time() - start_time
                elapsed_time += analysis_time
                print(f"Analysis elapsed time: {analysis_time:.3f} s")

                cts = {}
                for class_name, heatmap in result.heatmap_dicts.items():
                    cts[class_name] = {"contours": None, "hierarchy": None}
                    contours, hierarchy = cv2.findContours(heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
                    contours_list = [contour.tolist() for contour in contours]
                    hierarchy_list = hierarchy.tolist()
                    cts[class_name]["contours"] = contours_list
                    cts[class_name]["hierarchy"] = hierarchy_list

                file_name = test_file.split('/')[-1]
                contour_file_name = ''.join(file_name.split('.')[:-1]) + '.json'
                # Results are saved for debugging
                with open(f'./tests/__test_run__/{contour_file_name}', 'w') as outfile:
                    json.dump(cts, outfile)
                with open(f'./tests/snapshots/{contour_file_name}') as f:
                    expected_json = json.load(f)

                self.assertEqual(cts, expected_json)

                mock_function.assert_called()
                progress_values = []
                for call_args in mock_function.call_args_list:
                    self.assertEqual(len(call_args.args), 1)
                    progress_values.append(call_args.args[0])

                for i, progress_value in enumerate(progress_values):
                    self.assertGreaterEqual(progress_value, 0)
                    self.assertLessEqual(progress_value, 1)

                    # check if progress values monotonically increase
                    if i == 0:
                        continue
                    self.assertLessEqual(progress_values[i-1], progress_values[i])

        print(f"Total analysis elapsed time: %.3f s" % elapsed_time)

        # test gpu memory leak
        torch.cuda.empty_cache()
        after_analysis_usage = torch.cuda.memory_allocated()
        self.assertEqual(after_analysis_usage, start_memory)
        breast_resection_analyzer.unload()
        after_unload_usage = torch.cuda.memory_allocated()
        self.assertEqual(after_unload_usage, 0)


if __name__ == "__main__":
    unittest.main()
