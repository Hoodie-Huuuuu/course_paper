from typing import *

from icecream import ic
import numpy as np
from skimage.segmentation import watershed, slic, felzenszwalb, quickshift
import bisect 

import numpy.typing as npt

import segmenter_service_pb2 as pb2
import segmenter_service_pb2_grpc as pb2_grpc

# нумерация всегда с единицы
def segmentation(
    image_repr,
    method: Literal["slic", "watershed", "quick_shift", "fwb"],
    **method_params,
):
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
    :return: numerated regions from 1, num of last region == count of regions
    """
    regions = None
    ic(method_params, type(method_params))

    if method == "slic":
        regions = slic(
            image_repr["lab"],
            n_segments=method_params["n_segments"],
            compactness=method_params["compactness"],
            sigma=method_params["sigma"],
            start_label=1,
        )

    elif method == "watershed":  # 1
        regions = watershed(
            image_repr["edges"],
            markers=method_params["n_segments"],
            compactness=method_params["compactness"],
        )  # 0.001

    elif method == "quick_shift":  # 0 # it's better use lab image
        regions = quickshift(
            image_repr["lab"],
            kernel_size=method_params["sigma"],
            max_dist=method_params["max_dist"],  # Чем выше, тем меньше кластеров.
            ratio=method_params["ratio"],
        )  # Чем больше тем меньше компактность

    elif method == "fwb":  # 0
        regions = felzenszwalb(
            image_repr["rgb"],
            scale=int(method_params["scale"]),
            sigma=method_params["sigma"],
            min_size=int(method_params["min_size"]),
        )  # scale = 400 #min_size = 50

    renumerate_from_1(regions)
    return regions, np.amax(regions)


def renumerate_from_1(regions):
    min_num = np.amin(regions)
    regions += 1 if min_num == 0 else 0
    return


def renumerate_regions(regions: np.ndarray, sorted_nums_of_pixels, start=0):
    """

    :param regions: массив с регионами
    :param sorted_nums_of_pixels: их отсортирвоанные номера
    :param start: с какого числа начать перенумерацию
    :return: номер последнего
    """
    if start > sorted_nums_of_pixels[0]:
        raise ValueError(
            "renumerating of regions will be correct only with 'start' <= min(sorted_nums_of_pixels)"
        )

    for j, num in enumerate(sorted_nums_of_pixels, start=start):
        regions[regions == num] = j
    return j


def additionally_split(
    img_crop: np.ndarray,
    zone_to_split: npt.NDArray[np.bool],
    old_marker_mask: npt.NDArray[np.bool],
    new_marker_mask: npt.NDArray[np.bool],
    n_segments: int = 5,
    numeration_start: int = 0,
):
    """

    :param img_crop:
    :param zone_to_split: маска, в пределах которой идет подразбиение
    :param old_marker_mask: метки старого маркера
    :param new_marker_mask: метки нового маркера
    :param n_segments: начальное число суперпикселей, на которое надо подразбить область
    :param numeration_start: номер, с которого нумеруются суперпиксели
    :return: (матрица с номерами суперпикселей, количество суперпикселей)
    """
    # нумерация с 0 или 1 неважно
    good_segmentation = False  # True когда маркеры в разных суперпикселях
    new_regions = None

    while not good_segmentation:
        print("\ndo additional segmentation")
        new_regions = slic(
            img_crop, n_segments=n_segments, compactness=5, sigma=1, start_label=1
        )
        new_regions = np.where(zone_to_split, new_regions, -1)

        new_pixels_under_old_marks = np.unique(
            np.where(old_marker_mask, new_regions, -1)
        )
        new_pixels_under_old_marks = new_pixels_under_old_marks[
            new_pixels_under_old_marks != -1
        ]
        print(f"new pixels under OLD marks {new_pixels_under_old_marks}")

        new_marks_in_curr_pixel = np.where(zone_to_split, new_marker_mask, 0)
        new_pixels_under_new_marks = np.unique(
            np.where(new_marks_in_curr_pixel != 0, new_regions, -1)
        )
        new_pixels_under_new_marks = new_pixels_under_new_marks[
            new_pixels_under_new_marks != -1
        ]
        print(
            f"new pixels under NEW marks in local numeration{new_pixels_under_new_marks}"
        )

        if (
            np.intersect1d(
                ar1=new_pixels_under_old_marks,
                ar2=new_pixels_under_new_marks,
                assume_unique=True,
            ).shape[0]
            == 0
        ):
            good_segmentation = True
        n_segments *= 2

    # перенумерация суперпикселей по порядку с нуля
    nums = np.unique(new_regions)  # важно что отсортированный
    new_superpixel_numbers_in_old = nums[nums != -1]
    renumerate_regions(
        regions=new_regions,
        sorted_nums_of_pixels=new_superpixel_numbers_in_old,
        start=0,
    )

    # перенумерация с минимально возможного номера, чтобы не было конфликтов с уже существующими суперпикселями
    new_regions += numeration_start
    return new_regions, new_superpixel_numbers_in_old.size


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


def array_from_Mask2D(mask2d):
    width, height = mask2d.width, mask2d.height
    data = mask2d.data
    dtype = mask2d.dtype

    return np.frombuffer(data, dtype=dtype).reshape((height, width))


def array_from_NdArray(arr):
    return np.frombuffer(arr.data, dtype=arr.dtype).reshape(arr.shape)


def in_sorted_list(elem, sorted_list):
    i = bisect.bisect_left(sorted_list, elem)
    return i != len(sorted_list) and sorted_list[i] == elem