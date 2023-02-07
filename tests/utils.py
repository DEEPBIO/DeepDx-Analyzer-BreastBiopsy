import math
import numpy as np
from PIL import Image
import cv2

def isNdarrayEqual(array1: np.ndarray, array2: np.ndarray):
    shape1 = array1.shape
    shape2 = array2.shape
    if shape1 != shape2:
        return False
    return np.array_equal(array1, array2)

def isNdarrayEqualWithinTol(array1: np.ndarray, array2: np.ndarray):
    shape1 = array1.shape
    shape2 = array2.shape
    if shape1 != shape2:
        return False

    diff_indices = np.where((array1 == array2) == False)
    if len(shape1) == 3:
        for i in range(len(diff_indices[0])):
            i1, i2, i3 = diff_indices[0][i], diff_indices[1][i], diff_indices[2][i] 
            val1 = array1[i1,i2,i3]
            val2 = array2[i1,i2,i3]
            if not math.isclose(val1, val2, rel_tol=1e-09):
                return False
            # For debugging:
            # print(f'{val1}, {val2}')
            # print(f'{val1.as_integer_ratio()}, {val2.as_integer_ratio()}')
            # print(f'{val1.hex()}, {val2.hex()}')
    elif len(shape1) == 2:
        for i in range(len(diff_indices[0])):
            i1, i2 = diff_indices[0][i], diff_indices[1][i]
            val1 = array1[i1,i2]
            val2 = array2[i1,i2]
            if not math.isclose(val1, val2, rel_tol=1e-09):
                return False
            # For debugging:
            # print(f'{val1}, {val2}')
            # print(f'{val1.as_integer_ratio()}, {val2.as_integer_ratio()}')
            # print(f'{val1.hex()}, {val2.hex()}')

    return True

def hash_file_with_sha1(file_path):
    import sys
    import hashlib

    BUF_SIZE = 134217728  # 16MiB

    sha1 = hashlib.sha1()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()

def visualize_numpy_array(numpy_array, name = False):
    array = np.copy(numpy_array)
    array_max = np.amax(array)
    if array_max > 1e-9:
        array = (array / array_max * 255).astype(int)

    w, h = np.shape(array)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:,:,0] = np.transpose(array, [1, 0])
    
    img = Image.fromarray(img, 'RGB')
    if name:
        img.save(name)
    else:
        img.show()
    return img


def compare_images(input_image, output_image):
  # compare image dimensions (assumption 1)
    if input_image.size != output_image.size:
        return False

    rows, cols = input_image.size

  # compare image pixels (assumption 2 and 3)
    for row in range(rows):
        for col in range(cols):
            input_pixel = input_image.getpixel((row, col))
            output_pixel = output_image.getpixel((row, col))
            if input_pixel != output_pixel:
                return False
    return True

def save_heatmap_as_image(heatmap, name):
    arr = np.copy(heatmap)
    arr = np.transpose(arr, (2,1,0))

    for i in range(np.shape(heatmap)[0]):
        visualize_numpy_array(arr[:,:,i], f'{name}_{i}.png')

def load_np_dict(path):
    loaded_dict = {}
    data = np.load(path, allow_pickle=True)
    for key, item in data.item().items():
        loaded_dict[key] = item
    return loaded_dict


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