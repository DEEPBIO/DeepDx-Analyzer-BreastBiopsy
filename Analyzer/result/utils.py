from typing import Tuple
import PIL.Image
import cv2
import numpy as np
from openslide import OpenSlide

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


def get_region_mask(region, slide_size) -> np.ndarray:
    '''
    get region mask. Note that the size of shape of returned region mask is ((slide_w, slide_h))
    '''
    REGION_MASK_FILLED_VAL = 255

    slide_w, slide_h = slide_size
    mask = np.zeros((slide_h, slide_w), dtype=np.uint8)
    if region['type'] == 'rectangle':
        p1 = region['data']['min']
        p2 = region['data']['max']
        angle = region['data']['angle']

        if angle == 0:
            cv2.rectangle(mask, tuple(p1), tuple(p2), REGION_MASK_FILLED_VAL, -1)
        else:
            rect = (((p1[0]+p2[0])/2., (p1[1]+p2[1])/2.), (p2[0]-p1[0], p2[1]-p1[1]), angle)
            try:
                box = np.int0(cv2.boxPoints(rect))
            except:
                box = np.int0(cv2.cv.BoxPoints(rect))
            cv2.drawContours(mask, [box], -1, REGION_MASK_FILLED_VAL, -1)

    elif region['type'] == 'ellipse':
        p1 = region['data']['min']
        p2 = region['data']['max']
        angle = region['data']['angle']
        rect = (((p1[0]+p2[0])/2., (p1[1]+p2[1])/2.), (p2[0]-p1[0], p2[1]-p1[1]), angle)
        cv2.ellipse(mask, rect, REGION_MASK_FILLED_VAL, -1)

    elif region['type'] == 'contour':
        ptrs = region['data']
        ctrs = [np.array([[ptr] for ptr in k]) for k in ptrs]
        cv2.drawContours(mask, ctrs, -1, REGION_MASK_FILLED_VAL, -1)

    else:
        raise ValueError('unknown type %s' % region['type'])

    return mask.T