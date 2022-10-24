from typing import *
from PIL import Image
import numpy as np
from skimage.segmentation import watershed, slic, felzenszwalb, quickshift
from tkinter import ttk, Frame, Button, Label


# нумерация всегда с единицы
def init_segmentation(image_repr,
                      method: Literal["slic", "watershed", "quick_shift", "fwb"],
                      **method_params):
    # sigma=0, compactness=5, n_segments=600, ratio=0.5, max_dist=6,
    # min_size=50):
    """
    :param image_repr: dictionary of image representations
                      Any image if method == "slic"
                      grayscale edges if method == "watershed"
                      rgb image if method == "quick_shift" or "fwb"
    :param method: "slic", "watershed", "quick_shift", "fwb" - Felzenszwalb
    :param sigma: the Gaussian kernel
    :param compactness: compactness for slic and watershed methods
    :param n_segments: number of segments for slic and watershed

    Quick shift parameters
    :param ratio: Balances color-space proximity and image-space proximity.
                  Higher values give more weight to color-space.
    :param max_dist: Cut-off point for data distances. Higher means fewer clusters.

    Felzenszwalb parameters
    :param min_size: Minimum component size. Enforced using postprocessing.
    :return: regions, num of last region
    """
    regions = None
    print(method_params, type(method_params))

    if method == "slic":
        regions = slic(image_repr['lab'],
                       n_segments=method_params['n_segments'],
                       compactness=method_params['compactness'],
                       sigma=method_params['sigma'],
                       start_label=1)

    elif method == "watershed":  # 1
        regions = watershed(image_repr['edges'],
                            markers=method_params['n_segments'],
                            compactness=method_params['compactness'])  # 0.001

    elif method == "quick_shift":  # 0 # it's better use lab image
        regions = quickshift(image_repr['lab'],
                             kernel_size=method_params['sigma'],
                             max_dist=method_params['max_dist'],  # Чем выше, тем меньше кластеров.
                             ratio=method_params['ratio'])  # Чем больше тем меньше компактность

    elif method == "fwb":  # 0
        regions = felzenszwalb(image_repr['lab'],
                               scale=method_params['scale'],
                               sigma=method_params['sigma'],
                               min_size=method_params['min_size'])  # scale = 400 #min_size = 50

    renumerate_from_1(regions)
    return regions, np.amax(regions)


def renumerate_from_1(regions):
    min_num = np.amin(regions)
    regions += 1 if min_num == 0 else 0

# crop_height = ymax - ymin
# crop_width = xmax - xmin
#
# extended_ymin, extended_ymax = ymin - crop_height // 3, ymax + crop_height // 3
# extended_xmin, extended_xmax = xmin - crop_width // 3, xmax + crop_width // 3
#
# ymin = max(extended_ymin, 0)
# ymax = min(extended_ymax, height - 1)
# xmin = max(extended_xmin, 0)
# xmax = min(extended_xmax, width - 1)
