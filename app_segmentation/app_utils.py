import numpy as np
from icecream import ic
from PIL import Image
from typing import *
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte, img_as_float
import segmenter_service_pb2 as pb2
import segmenter_service_pb2_grpc as pb2_grpc


def get_image(mask: np.ndarray, source_image: np.ndarray, colors_rgb: Dict[int, tuple[int,int,int]], alpha=0.5,
              draw_borders=False, regions=None):
    if draw_borders:
        if regions is None:
            raise ValueError("regions must be passed if draw_borders==True")
        source_image = img_as_ubyte(mark_boundaries(img_as_float(source_image), regions))
        ic(type(source_image), source_image.dtype, np.amax(source_image))
    mask_rgb = _get_rgb_mask(mask_template=mask, colours_rgb=colors_rgb)
    return _make_image(mask, image_rgb=source_image, mask_rgb=mask_rgb, alpha=alpha)

def _make_image(marker_mask_2d: np.ndarray, image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha=0.5):
    ic(alpha)

    if alpha > 1 or alpha < 0:
        raise ValueError

    img_ar = np.where(
        (marker_mask_2d != 0)[:, :, np.newaxis],
        mask_rgb * (1 - alpha) + alpha * image_rgb,
        image_rgb
    )

    img = Image.fromarray(np.clip(img_ar.astype(int), 0, 255).astype('uint8'), "RGB")
    return img

def _get_rgb_mask(mask_template: np.ndarray, colours_rgb: Dict[int, tuple[int,int,int]]):
    """
    :param mask_template: np.ndarray of shape (M, D)
    :return: rgb mask with shape (M, D, 3)
    """
    mask_rgb = np.zeros((*mask_template.shape, 3))
    for marker_index, marker_color in colours_rgb.items():
        mask_rgb[mask_template == marker_index] = marker_color
    return mask_rgb


def Mask2D_to_array(mask: pb2.Mask2D) -> np.ndarray:
    return np.frombuffer(mask.data, dtype=mask.dtype).reshape((mask.height, mask.width))