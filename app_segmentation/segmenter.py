from typing import *

import cv2 as cv
import numpy as np
from PIL import Image, ImageColor
from skimage import filters
from skimage.segmentation import watershed


class Segmenter:
    def __init__(self, image: Image, markers: Dict[str, str]):
        """
        :param image: RGB PIL.Image
        :param markers: OrderedDict["marker_name", "hex_color"]
        """

        # пароги для чувствительности
        self._thresholds = {"mean": 20,
                           "var": 20,
                           "radius": 100  # в пикселях
                            }
        # маркеры
        self._markers = markers

        # стек состояний маски
        self._previous_masks_stack = []

        # изображение в LAB координатах
        self._lab_image = cv.cvtColor(src=np.asarray(image), code=cv.COLOR_RGB2LAB)

        # список цветов в rgb
        hex2rgb = lambda hex_colour: ImageColor.getcolor(hex_colour, "RGB")
        self._colours_rgb = {i: hex2rgb(hex_color) for i, hex_color in self._markers.values()}

        # входное изображение (изначально без разметок)
        self._rgb_marked_image = image
        self._rgb_input_image_ar = np.asarray(image)
        self._gray_ar = np.asarray(image.convert('L'))

        # выходная маска сегментатора
        self._mask = np.zeros((image.height, image.width), dtype='uint8')

        # метод сегментации (watershed)
        edges = filters.sobel(self._gray_ar)
        self._regions = watershed(edges, markers=700, compactness=0.001)
        return


    # закрашивает суперпиксели в соответсвии с данными метками и чувствительностью
    def draw_regions(self, filled_mask: np.ndarray, marker_mask: np.ndarray = None,
                     curr_marker: str = "None", sens: float = 0, change_mask=True) -> np.array:
        """

        :param filled_mask: ndarray(image.height, image.width) - массив заполненный метками маркеров
        :param marker_mask: ndarray(image.height, image.width) - массив с последним штрихом пользователя
        :param curr_marker:  str - имя маркера, которым пользователь только что закончил рисовать
        :param sens: float - чувствительность
        :param change_mask: нужно ли менять маску сегментатора на ту, что нарисует этот метод
        :return метод возвращает результирующую маску
        """

        # та жа самая маска, но с цветами, для отображения
        mask_rgb = np.zeros((*filled_mask.shape, 3))
        for marker_index, marker_color in self._colours_rgb.items():
            mask_rgb[filled_mask == marker_index] = self._colours_rgb[marker_index]

        if curr_marker == "None" or marker_mask is None:
            self._rgb_marked_image = self._make_image(mask_rgb)
            if change_mask:
                self._mask = filled_mask.copy()
            return self._mask.copy()

        if filled_mask.shape != marker_mask.shape:
            raise ValueError("filled mask and marker mask must have the same shape")

        # результат
        res_mask = np.copy(filled_mask)
        curr_marker_idx = self._markers[curr_marker][0]
        # номера суперпикселей, на которые попал маркер
        marked_regions = np.unique(self._regions[marker_mask == curr_marker_idx])

        # для каждого суперпикселя, на который попал последний штрих
        # в радиусе сравниваем характеристики в соотвествии с чувствительностью
        processed_area = []
        for region_num in marked_regions:
            #  окно просморта -> номера суперпикселей, попавших в окно
            radius = int(self._thresholds['radius'] * sens)
            sps_around = self._sps_around(region_num, radius)
            properties = self._region_property(region_num)

            for superpixel_num in sps_around:
                if superpixel_num in processed_area:
                    break
                properties_superpixel = self._region_property(superpixel_num)

                all_prop_good = True
                for prop_name, prop_val in properties_superpixel.items():
                    if np.linalg.norm(properties[prop_name] - prop_val) > self._thresholds[prop_name] * sens:
                        all_prop_good = False
                        break

                if all_prop_good:
                    processed_area += superpixel_num
                    res_mask[self._regions == superpixel_num] = curr_marker_idx  # отметили суперпиксель
                    mask_rgb[self._regions == superpixel_num, ...] = self._colours_rgb[curr_marker_idx]


            res_mask[self._regions == region_num] = curr_marker_idx  # отметили суперпиксель
            mask_rgb[self._regions == region_num, ...] = self._colours_rgb[curr_marker_idx]

        # полупрозрачная маска на картинке
        if change_mask:
            self._mask = res_mask.copy()
        self._rgb_marked_image = self._make_image(mask_rgb)
        return res_mask

    # возвращет номера суперпикселей вокруг
    def _sps_around(self, region_num: int, radius: int):
        indexes = (np.where(self._regions == region_num))
        length = indexes[0].shape[0]
        indexes = list(zip(indexes[0], indexes[1]))
        center_idx = indexes[length // 2]
        y, x = center_idx

        height, width = self._regions.shape
        x1, y1, x2, y2 = x - radius, y - radius, x + radius, y + radius
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > width:
            x2 = width - 1
        if y2 > height:
            y2 = height - 1

        return list(np.unique(self._regions[y1: y2 + 1, x1: x2 + 1]))

    def _make_image(self, mask_rgb: np.ndarray, alpha=0.4):
        img_ar = np.where(mask_rgb, self._gray_ar[:, :, np.newaxis] * alpha, self._rgb_input_image_ar)
        img_ar = img_ar + (1 - alpha) * mask_rgb
        img = Image.fromarray(np.clip(img_ar.astype(int), 0, 255).astype('uint8'), "RGB")
        return img

    def _region_property(self, number: int) -> Dict[str, float]:
        mask = (self._regions == number)
        count = np.sum(mask)
        mean = np.sum(self._lab_image[mask, ...], axis=0) / count
        # mean = np.linalg.norm(mean)
        variance = np.sum((self._lab_image[mask, ...] - mean) ** 2, axis=0) / count
        # variance = np.linalg.norm(variance)
        return {"mean": mean, "var": variance}

    def push_state(self, mask: np.array, marker_mask: np.array, curr_marker, change_mask=False):
        if change_mask:
            self._mask = mask.copy()
        self._previous_masks_stack.append((mask.copy(), marker_mask.copy(), curr_marker))
        return

    def pop_state(self):
        if len(self._previous_masks_stack) == 0:
            return None
        state = self._previous_masks_stack.pop()
        self._mask = np.copy(state[0])
        return state

    @property
    def mask(self):
        return self._mask.copy()

    @property
    def rgb_marked_image(self):
        return self._rgb_marked_image

    def states_len(self):
        return len(self._previous_masks_stack)

    def get_state(self, idx: int = -1):
        return self._previous_masks_stack[idx]

    def get_user_marks(self) -> np.array:
        res = np.zeros(shape=self._mask.shape)
        for _, marker_mask, marker in self._previous_masks_stack:
            res = res + marker_mask
        return res







