import numpy as np
from icecream import ic
from PIL import Image
from typing import *


def get_image(mask: np.ndarray, source_image: np.ndarray, colors_rgb: Dict[int, tuple[int,int,int]], alpha=0.5):
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