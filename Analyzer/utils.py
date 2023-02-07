from typing import Tuple
import time
from logging import Logger
import torch
import PIL.Image
import cv2
import numpy as np
from openslide import OpenSlide
from .type import Region, Coordinate

find_contour_returns = cv2.findContours(np.ndarray(
    shape=(100, 100), dtype=np.int32), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

if len(find_contour_returns) == 3:
    def find_contours(image, mode, method, **kwargs):
        image, contours, hierarchy = cv2.findContours(
            image, mode, method, **kwargs)
        return contours, hierarchy
else:
    def find_contours(image, mode, method, **kwargs):
        return cv2.findContours(image, mode, method, **kwargs)


def get_mpp(slide: OpenSlide) -> Tuple[float, float]:
    try:
        mpp_x = float(slide.properties['openslide.mpp-x'])
        mpp_y = float(slide.properties['openslide.mpp-y'])
        return mpp_x, mpp_y
    except:
        try:
            mpp_x = 10000 / float(slide.properties['tiff.XResolution'])
            mpp_y = 10000 / float(slide.properties['tiff.YResolution'])
            return mpp_x, mpp_y
        except:
            raise ValueError("Cannot read mpp")


def log_elapsed(logger: Logger):
    def log_decorator(function):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = function(*args, **kwargs)
            logger.debug(" [elapsed] %s: %.3f s", function.__name__, time.time() - start_time)
            return result
        return wrapper
    return log_decorator

def printMem(a:str):
    print(a)
    print(f"{torch.cuda.memory_allocated()} {torch.cuda.max_memory_allocated()}")
    print(f"{torch.cuda.memory_reserved()} {torch.cuda.max_memory_reserved()}")
    print("---------------")

def get_bounding_box(region: Region, dimensions) -> Tuple[Coordinate, Coordinate]:
    width, height = dimensions
    if region['type'] in ('rectangle', 'ellipse'):
        p1 = region['data']['min']
        p2 = region['data']['max']
        angle = region['data']['angle']
        if angle == 0:
            upper, lower = p1, p2
        else:
            rect = (((p1[0]+p2[0])/2., (p1[1]+p2[1])/2.), (p2[0]-p1[0], p2[1]-p1[1]), angle)
            try:
                box = np.int0(cv2.boxPoints(rect))
            except:
                box = np.int0(cv2.cv.BoxPoints(rect))
            upper, lower = list(box.min(axis=0)), list(box.max(axis=0))
    elif region['type'] == 'contour':
        ptrs = region['data']
        flat = []
        for seq in ptrs:
            flat += seq
        outside = np.array(flat)
        upper, lower = list(outside.min(axis=0)), list(outside.max(axis=0))
    else:
        raise ValueError('unknown type %s' % region['type'])

    if upper[0] < 0:
        upper[0] = 0
    if upper[1] < 0:
        upper[1] = 0
    if width < lower[0]:
        lower[0] = width - 1
    if height < lower[1]:
        lower[1] = height - 1

    return upper, lower

def get_level(slide_path):
    """
        This code returns frequently used variables.

        Args:
            slide_path: path to slide.

        Returns:
            lv_index : Return the slide level when mpp is 2 or 8
            raw_mpp_x : Returns the mpp of WSI.
            scale : It is a multiplier to adjust the mpp of wsi to 1.
            patch_size : The size of the patch used to divide the gird. It is 512 in size based on mpp 1.
            dimensions : size of wsi when mpp is 1
            downsamples : downsample rates for each lv_index
    """
    #get mpp of wsi
    slide = OpenSlide(slide_path)
    raw_mpp = get_mpp(slide)

    lv_index = [0, 0]
    lv_list = [2, 8]

    for _, lv in enumerate(lv_index):
        try:
            while True:
                level_mpp = raw_mpp[0] * slide.level_downsamples[lv_index[_]+1]
                if (level_mpp > 0.8 * lv_list[_]):#지정한 mpp보다 높아지는 경우에
                    break
                lv_index[_] = lv_index[_] + 1
        except:
            lv_index[_] = 0

    downsamples = [slide.level_downsamples[lv_index[0]], slide.level_downsamples[lv_index[1]]]
    #scale is a value to set the mpp of the slide to 1
    scale = 1 / raw_mpp[0]

    """
        patch_size is 512x512 pixels based on mpp1, so it is 2048x2048 based on mpp 0.25.
        In the same way, change the dimension according to mpp1.
    """
    patch_size = int(512 * scale)
    dimensions = np.array(slide.level_dimensions[0])/scale
    dimensions = dimensions.astype(int)

    return lv_index, raw_mpp, scale, patch_size, dimensions, downsamples

def get_mpp(slide: OpenSlide) -> Tuple[float, float]:
    try:
        mpp_x = float(slide.properties['openslide.mpp-x'])
        mpp_y = float(slide.properties['openslide.mpp-y'])
        return mpp_x, mpp_y
    except:
        try:
            mpp_x = 10000 / float(slide.properties['tiff.XResolution'])
            mpp_y = 10000 / float(slide.properties['tiff.YResolution'])
            return mpp_x, mpp_y
        except:
            raise ValueError("Cannot read mpp")