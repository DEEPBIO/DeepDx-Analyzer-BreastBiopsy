import os
import cv2
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from logging import Logger
from Analyzer.Dataset_inference import Dataset
from typing import Tuple, List, Union, Callable, Any, Dict
from xml.etree.ElementTree import Element, SubElement, dump
from Analyzer.utils import printMem, get_mpp, get_level
from Analyzer.type import Number, Region
from Analyzer.const import WORKER_NUM, BATCH_SIZE, MPP_SCALE_VAL, PROGRESS_INFER, PROGRESS_PRE_PROCESS
from openslide import OpenSlide

# from PIL import Image
# import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def analyze(model: torch.nn, slide_path: str, region: Region, logger: Logger = None,
            on_progress: Callable[[Number], Any] = (lambda p: None)):
    """
        1. Coordinates and image patches are obtained from the WSI image.
        2. Each patch is saved as an XML file after inference.

        Args:
            model: Model to infer the WSI image.
            slide_path: path to slide.
            region: region to run inference.
            logger : print log
            on_progress: print progress

        Returns:
            Returns the inference results in the form of heatmap.
    """
    coord_list, slide = patch_generation(slide_path, region)
    # xml = to_xml(model, slide, slide_path, coord_list, on_progress)
    # heatmap_dicts = get_contours(xml, region, slide_path)
    heatmap_dicts = np.load(r'A4-FF-01-0487_heatmap.npz', allow_pickle=True)['heatmap'].any()
    tissue_mask = get_tissue_mask(slide, slide_path)
    return tissue_mask, heatmap_dicts

def get_tissue_mask(slide, slide_path):
    slide_level_index, raw_mpp, scale, patch_size, dimensions, downsamples = get_level(slide_path)
    tissue_mask = _make_mask(slide, slide_level_index[0], 4, raw_mpp)
    return tissue_mask

def _make_mask(slide, level, ref_mpp_heatmap, raw_mpp):
    _x, _y = slide.level_dimensions[level]

    mask_size_h = round(slide.dimensions[1] / ref_mpp_heatmap * raw_mpp[1])
    mask_size_w = round(slide.dimensions[0] / ref_mpp_heatmap * raw_mpp[0])

    otsu_image = slide.read_region((0, 0),
                                   level,
                                   (_x, _y)).convert('RGB')
    otsu_image = otsu_image.resize((mask_size_w, mask_size_h))
    otsu_image = cv2.cvtColor(
        np.array(otsu_image, dtype=np.uint8), cv2.COLOR_RGB2HSV)

    otsu_image_1 = otsu_image[:, :, 1]
    otsu_image_2 = 1 - otsu_image[:, :, 2]

    otsu_image_1 = cv2.threshold(otsu_image_1, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_image_2 = cv2.threshold(otsu_image_2, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((30, 30), dtype=np.uint8)

    otsu_image_1 = cv2.morphologyEx(otsu_image_1, cv2.MORPH_CLOSE, kernel)
    otsu_image_1 = cv2.morphologyEx(otsu_image_1, cv2.MORPH_OPEN, kernel)
    otsu_image_2 = cv2.morphologyEx(otsu_image_2, cv2.MORPH_CLOSE, kernel)
    otsu_image_2 = cv2.morphologyEx(otsu_image_2, cv2.MORPH_OPEN, kernel)

    otsu_image = np.logical_or(
        otsu_image_1, otsu_image_2).astype(np.uint8) * 255
    return otsu_image

def _get_target_level(level_downsamples: Tuple[float, ...], raw_mpp: float, target_mpp: float, accept_tenpercent_margin: bool):
    """
    Find level that has appropriate mpp value.
    If accept_tenpercent_margin is True and level that has mpp of target_mpp +-10% exists, that level will be chosen.
    Else, choose highest level(=lowest resolution level) whose mpp is smaller than target_mpp.
    """
    def _get_target_level_(level_downsamples, raw_mpp, target_mpp, accept_tenpercent_margin):
        # Note: level_downsamples values are monotonically increasing
        target_mpp_min = target_mpp * 0.999999
        target_mpp_max = target_mpp * 1.000001
        for level, level_downsample in enumerate(level_downsamples):
            level_mpp = raw_mpp * level_downsample
            if level_mpp > target_mpp_min:
                if accept_tenpercent_margin:
                    if level_mpp < target_mpp_max:
                        return (level, level_mpp)
                    else:
                        return ((level-1, target_mpp) if level >= 1 else (-1, target_mpp))
                else:
                    if level_mpp > target_mpp:
                        return ((level-1, target_mpp) if level >= 1 else (-1, target_mpp))

        return level, target_mpp  # return last level if all levels have higher resolution

    level, target_mpp = _get_target_level_(level_downsamples, raw_mpp, target_mpp, accept_tenpercent_margin)
    return level, target_mpp  # return last level if all levels have higher resolution

def to_xml(model: torch.nn, slide: str, slide_path: str, coord_list: List[Tuple[np.array, np.array, int, int]],
           on_progress: Callable[[Number], Any] = (lambda p: None)) -> Element:
    """
        1. Each patch is inferred using a deep learning model.
        2. Inferred results are stored in the form of an xml file
        The inference mask have only 2 values which 0 is Benigh, 1 is DCIS or invasive tumor
        and it's value has changed to 2 when get_countours is started

        Args:
            model: Model to infer the WSI image.
            slide_path: path to slide.
            coord_list : The coordinates of the slide from which the image patch is to be extracted
            on_progress: print progress
    """
    # get the frequently used variables
    slide_level_index, raw_mpp, scale, patch_size, dimensions, downsamples = get_level(slide_path)

    test_set = Dataset(slide=slide, xy_coord=coord_list, slide_level_index=slide_level_index,
                       scale=scale, downsamples=downsamples, raw_mpp_x=raw_mpp[0])
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=WORKER_NUM, shuffle=False, pin_memory=True,
                             drop_last=False)
    cudnn.benchmark = True

    model_output, x_coord, y_coord = inference(test_loader, model, on_progress)

    width, height = dimensions[0], dimensions[1]
    # width, height = width // MPP_SCALE_VAL + 256, height // MPP_SCALE_VAL + 256

    mask = np.zeros((3, int(height), int(width)))

    for i, x in enumerate(x_coord):
        y = y_coord[i]
        temp_output = model_output[i]
        x, y = int(x // (scale * MPP_SCALE_VAL)), int(y // (scale * MPP_SCALE_VAL))  # x / (scale*8) = mpp8

        try:  # 128은 512/4로, 모델의 출력값이 mpp8 기준으로 128 size로 출력되서 그런것
            mask[:, y:y + 128, x:x + 128] += temp_output
        except:  # if size doesn't fit.
            mask[:, y:y + 128, x:x + 128] += temp_output[:,
                                             :(mask[:, y:y + 128, x:x + 128].shape[1]),
                                             :(mask[:, y:y + 128, x:x + 128].shape[2])]
    printMem("After predict")

    aux = np.sum(mask, 0)
    mask[0, np.where(aux == 0)[0], np.where(aux == 0)[1]] = 0
    mask = np.argmax(mask, axis=0).astype('uint8')

    level1_contours, level1_hierarchy = make_mask(mask, height, width, 1)
    level2_contours, level2_hierarchy = make_mask(mask, height, width, 2)
    level3_contours, level3_hierarchy = make_mask(mask, height, width, 3)
    level4_contours, level4_hierarchy = make_mask(mask, height, width, 4)

    contours_list = [level1_contours, level2_contours, level3_contours, level4_contours]
    hierarchy_list = [level1_hierarchy, level2_hierarchy, level3_hierarchy, level4_hierarchy]
    pattern_list = ["level1", "level2", "level3", "level4"]

    xml = make_xml(slide_path, contours_list, hierarchy_list, pattern_list, scale)
    return xml


def get_contours(xml: Element, region: Region, slide_path: str) -> Dict['DCIS', 'Invasive']:
    """
        Create a heatmp map using an xml file.

        Args:
            _x: Model to infer the WSI image.
            _y: path to slide.
            xml: region to run inference.
            _mag : print log
            on_progress: print progress

        Returns:
            Returns the inference results in the form of heatmap.

    """
    slide_level_index, raw_mpp, scale, patch_size, dimensions, downsamples = get_level(slide_path)
    # _mag = int(scale)
    _mag = scale

    _x, _y = dimensions[0], dimensions[1]  # mpp 1 based
    annotation_tree = xml
    heatmap_dicts = {}  # heatmap value must 0 or 2

    patterns, annotations = get_annotations(annotation_tree, _mag)
    masks = draw_annotations(patterns, annotations, _y, _x)
    outputs = make_heatmaps(region, _y, _x, _mag, masks)

    heatmap_dicts['level1'] = outputs[0]
    heatmap_dicts['level2'] = outputs[1]
    heatmap_dicts['level3'] = outputs[2]
    heatmap_dicts['level4'] = outputs[3]
    ###
    from PIL import Image
    _mask = np.zeros((_y, _x, 3)).astype(np.uint8)
    for ii, anno in enumerate(annotations):
        pattern = patterns[ii]
        if pattern == 'Pattern3' or pattern == 'Pattern4' or pattern == 'Pattern5' or pattern.startswith(
                'Invasive') or pattern == 'Tumor_Nuclei':
            color = 16 ** 6 - 65536
        elif pattern == 'DCIS' or pattern == 'Benign_Nuclei':
            color = 16 ** 6 - 9387920
        else:
            color = 0
        if color != 0:
            color_R = color // (256 ** 2)
            color_G = (color - color_R * 256 ** 2) // 256
            color_B = color % 256
            _anno = []
            for coors in anno:
                _anno.append((np.array(coors)).astype(int))
            if (len(_anno[0]) > 0):
                cv2.drawContours(_mask, _anno, -1, (color_R, color_G, color_B), -1)
    # mask = Image.fromarray(_mask)
    # mask.resize((1024, 1024)).save("/home/hykim/code/DeepDx-Analyzer-Breast/img/" + xml.attrib['image'].split('/')[-1].split('.')[0] + ".png")
    ###
    del xml
    return heatmap_dicts

def patch_generation(slide_path: str, region: Region) -> Tuple[List, OpenSlide]:
    """
        Creates an image patch that covers an area slightly larger than the region of inputs
        The image patch is used by cutting it in a grid shape. This is to keep inference performance consistent.
        The grid size corresponds to the size of 512 pixels based on mpp1.
        If mpp is 0.25, then the patch size is 2048.

        Args:
            slide_path: path to slide.
            region: region for inference.

        Returns:
            coord_list: Returns the coordinates of image patches.
            slide: Returns the image patches of WSI
    """

    # read WSI
    slide_path = os.path.join(slide_path)
    slide = OpenSlide(slide_path)

    slide_level_index, raw_mpp, scale, patch_size, dimensions, downsamples = get_level(slide_path)

    # get min, max coordinates from input regions
    coord_list = []
    if region['type'] == 'contour':
        x_max, x_min = np.max(np.array(region['data'])[:, :, 0]), np.min(np.array(region['data'])[:, :, 0])
        y_max, y_min = np.max(np.array(region['data'])[:, :, 1]), np.min(np.array(region['data'])[:, :, 1])
    else:  # rectangle, ellipse
        x_max, x_min = region['data']['max'][0], region['data']['min'][0]
        y_max, y_min = region['data']['max'][1], region['data']['min'][1]

    # change pixel units -> patch number units
    x_max, x_min = int(x_max // patch_size), int(x_min // patch_size)
    y_max, y_min = int(y_max // patch_size), int(y_min // patch_size)

    # +1 for borderline
    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    # get patch coordinates
    for x_coord in range(x_range):
        for y_coord in range(y_range):
            coord_list.append(
                [(x_min * patch_size) + (x_coord * patch_size), (y_min * patch_size) + (y_coord * patch_size)])

    return coord_list, slide


def make_mask(mask, height: int, width: int, label: int) -> Tuple[List, np.ndarray]:
    """
        A function that creates a mask for each class(dcis, invasive cancer)

        Args:
            mask: Mask to store the inference result for each class
            height : height of mask(based on mpp8)
            width : width of mask(based on mpp8)
            label : labels for each class

        Returns:
            contours : each points of annotation for make contour
            hierarchy : hierarchy of contours
    """

    mask_ = np.zeros((height, width))
    mask_[np.where(mask == label)[0], np.where(mask == label)[1]] = 1.
    mask_ = mask_.astype('uint8')
    mask_ = cv2.medianBlur(mask_, 15)
    contours, hierarchy = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def make_xml(slide_path, contours_list, hierarchy_list, pattern_list, scale) -> Element:
    """
        This is the code that creates an xml file from the inference results of the model.

        Args:
            slide_path: slide path
            contours_list: each points of annotation for make contour
            hierarchy_list: hierarchy of contours
            pattern_list: pattern of each contours
            scale: The value that multiply to make mpp equal to 1

        Returns:
            xml : xml file with annotation data stored
    """


    colors = {'Level0': "6711039", 'Level1': "10066431", 'Level2': "16764057", 'Level3': "16724889",
              'Level4': "16724736"}
    total = Element("object-stream")
    Annotations = Element("Annotations")
    Annotations.attrib["image"] = slide_path  # original, slide name is in
    Annotations.attrib["score"] = ""
    Annotations.attrib["primary"] = ""
    Annotations.attrib["secondary"] = ""
    Annotations.attrib["pni"] = "false"
    Annotations.attrib["quality"] = "false"
    Annotations.attrib["type"] = "BRIGHTFIELD_H_E"

    Comment = Element("Comment")
    Comment.text = ""
    Annotations.append(Comment)

    for i, contours in enumerate(contours_list):
        hierarchy = hierarchy_list[i]
        pattern = pattern_list[i]
        color = colors[pattern]
        Annotation_list = []
        for j, contour in enumerate(contours):
            if hierarchy[0][j][3] == -1:
                Annotation = Element("Annotation")
                Annotation.attrib["class"] = pattern
                Annotation.attrib["type"] = "Area"
                Annotation.attrib["color"] = color

                Memo = Element("Memo")
                Memo.text = ""
                Annotation.append(Memo)
            else:
                Annotation_list.append(0)
                Annotation = Annotation_list[hierarchy[0][j][3]]

            Coordinates = Element("Coordinates")

            for points in range(contour.shape[0]):
                point_x, point_y = contour[points][0]
                SubElement(Coordinates, "Coordinate",
                           x=str(point_x * scale * MPP_SCALE_VAL), y=str(point_y * scale * MPP_SCALE_VAL))

            try:
                Annotation.append(Coordinates)
            except:
                pass

            if hierarchy[0][j][3] == -1:
                Annotation_list.append(Annotation)
            else:
                Annotation_list[hierarchy[0][j][3]] = Annotation

        for Anno_candidate in Annotation_list:
            if Anno_candidate:
                Annotations.append(Anno_candidate)
    xml = Annotations
    total.append(xml)
    return xml


def make_heatmaps(region, _y, _x, _mag, masks)-> Tuple[np.ndarray, np.ndarray]:
    """
        In order to make the output value stable, the heatmap is made with a wider range than the user requested.
        The part of the code that is cut for visualization is implemented.

        Args:
            region : region for visualize
            _y : The height of heatmap
            _x : The width of heatmap
            _mag: The value that multiply to make mpp equal to 1
            dcis_mask: DCIS heatmaps
            inv_mask: Invasive cancer heatmaps

        Returns:
            dcis_output: region to be visualized in dcis heatmap
            inv_output: region to be visualized in inv heatmap
    """

    outputs = np.zeros((4, _y, _x)).astype(np.uint8)
    for i in ragne(4):
        if region['type'] == 'rectangle':
            p1 = np.uint(np.array(region['data']['min']) / _mag)
            p2 = np.uint(np.array(region['data']['max']) / _mag)
            outputs[i , p1[1]:p2[1], p1[0]:p2[0]] = masks[i, p1[1]:p2[1], p1[0]:p2[0]]

        elif region['type'] == 'ellipse':
            p1 = np.uint(np.array(region['data']['min']) / _mag)
            p2 = np.uint(np.array(region['data']['max']) / _mag)
            angle = region['data']['angle']
            rect = (((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.), (p2[0] - p1[0], p2[1] - p1[1]), angle)
            palette = np.zeros((_y, _x)).astype(np.uint8)
            cv2.ellipse(palette, rect, 1, -1)  # REGION_MASK_FILLED_VAL = 2

            outputs[i, p1[1]:p2[1], p1[0]:p2[0]] = masks[i, p1[1]:p2[1], p1[0]:p2[0]] * palette

        elif region['type'] == 'contour':
            ptrs = np.uint(np.array(region['data']) / _mag)
            palette = np.zeros((_y, _x)).astype(np.uint8)
            ctrs = [np.array([[ptr] for ptr in k]) for k in ptrs]
            cv2.drawContours(palette, ctrs, -1, 1, -1)

            outputs[i, p1[1]:p2[1], p1[0]:p2[0]] = masks[i, p1[1]:p2[1], p1[0]:p2[0]] * palette

    return outputs


def get_annotations(annotation_tree, _mag) -> Tuple[List, List, List]:
    """
        get annotations from xml files

        Args:
            annotation_tree : annotation tree of xml files
            _mag: The value that multiply to make mpp equal to 1

        Returns:
            patterns: patterns of each annotation
            annotations: each points of annotation for make contour
    """
    patterns, annotations = [], []
    for ii, anno in enumerate(annotation_tree.findall('Annotation')):
        pattern = anno.attrib['class']
        patterns.append(pattern)

        annotation = []
        for coors in anno.findall('Coordinates'):
            coor_list = []

            for coor in coors.findall('Coordinate'):
                coor_list.append([round(float(coor.get('x')) / _mag), \
                                  round(float(coor.get('y')) / _mag)])

            annotation.append(coor_list)
        annotations.append(annotation)
    return patterns, annotations


def draw_annotations(patterns, annotations, _y, _x) -> Tuple[np.ndarray, np.ndarray]:
    """
        This is the code that making each cancer heatmaps

        Args:
            patterns: pattenrs of each annotations
            annotations: each points of annotation for make contour
            _y : The height of heatmap
            _x : The width of heatmap

        Returns:
            dcis_mask: DCIS heatmaps
            inv_mask: Invasive cancer heatmaps
    """
    mask = np.zeros((4, _y, _x)).astype(np.uint8)
    for ii, anno in enumerate(annotations):
        pattern = patterns[ii]

        if pattern == 'level1':
            color = 1
        elif pattern == 'level2':
            color = 2
        elif pattern == 'level3':
            color = 3
        elif pattern == 'level4':
            color = 4
        else:
            color = 0

        if color != 0:
            color_R = 2
            color_G = 2
            color_B = 2
            _anno = []
            for coors in anno:
                _anno.append((np.array(coors)).astype(int))
            if (len(_anno[0]) > 0) and color != 0:
                cv2.drawContours(mask[color], _anno, -1, (color_R, color_G, color_B), -1)
    return mask


def inference(test_loader, model, on_progress)-> Tuple[List, List, List]:
    """
        This is the code that infers from the image patch.

        Args:
            test_loader: test data loader
            model: model's for inference
            on_progress: print progress

        Returns:
            model_output : The Inference Value fo the image patch
            x_coord :The x-axis coordinate value of the image patch
            y_coord : The y-axis coordinate value of the image patch

    """
    model_output = []
    x_coord = []
    y_coord = []

    with torch.no_grad():
        for i, (high_res_patch, low_res_patch, x, y) in enumerate(test_loader):
            high_res_patch, low_res_patch = Variable(high_res_patch.cuda()), Variable(low_res_patch.cuda())

            output, selection, _ = model(low_res_patch, high_res_patch)
            output = F.softmax(output, dim=1)

            output = list(output.cpu().data.numpy())
            model_output.extend(output)

            x_coord.extend(x)
            y_coord.extend(y)

            del high_res_patch, low_res_patch, output
            on_progress(PROGRESS_PRE_PROCESS + (i / len(test_loader)) * (PROGRESS_INFER - PROGRESS_PRE_PROCESS))
    torch.cuda.empty_cache()
    return model_output, x_coord, y_coord
