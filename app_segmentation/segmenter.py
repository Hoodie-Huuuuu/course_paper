from typing import *

import cv2 as cv
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageColor
from skimage import filters
from skimage.segmentation import watershed, slic, felzenszwalb, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from dataclasses import dataclass


# todo исключения если рисовать у края картинки
# лучше разбивать на felzenszwalb в начале, потом подразбивать на watershed или slic


@dataclass
class Point:
    x: int
    y: int


class Segmenter:
    def __init__(self, image: Image, markers: Dict[str, str]):
        """
        :param image: RGB PIL.Image
        :param markers: OrderedDict["marker_name", "hex_color"]

        Сначала вызывается метод draw
        отрисованное изображение доступно через свойство rgb_marked_image
        """

        # пароги для чувствительности
        self._thresholds = {"mean": 50,
                            "var": 50,
                            "radius": 100  # в пикселях
                            }
        # маркеры
        self._markers = markers

        # стек состояний маски
        # на любом шаге лежит (предыдущая маска, новые штрихи на этой маске, номермаркера, деление на суперпиксели)
        self._states_stack = []

        # изображение в LAB координатах
        self._lab_image = cv.cvtColor(src=np.asarray(image), code=cv.COLOR_RGB2LAB)

        # список цветов в rgb
        hex2rgb = lambda hex_colour: ImageColor.getcolor(hex_colour, "RGB")
        self._colours_rgb = {i: hex2rgb(hex_color) for i, hex_color in self._markers.values()}

        # входное изображение (изначально без разметок)
        self._rgb_marked_image = image
        self._rgb_input_image_ar = np.asarray(image)

        # выходная маска сегментатора
        self._mask = np.zeros((image.height, image.width), dtype='uint8')
        # все штрихи пользователя
        self._user_marks = np.zeros((image.height, image.width), dtype='uint8')

        # ======= метод сегментации ========
        # Watershed
        self._gray_ar = np.asarray(image.convert('L'))
        self._edges = filters.sobel(self._gray_ar)
        # self._regions = watershed(self._edges, markers=600, compactness=0.001)

        # Quickshift
        # self._regions = quickshift(self._rgb_input_image_ar, kernel_size=3, max_dist=6, ratio=0.5)

        # Felzenszwalb
        self._regions = felzenszwalb(self._rgb_input_image_ar, scale=400, sigma=0.5, min_size=50)
        # =================================== Image.fromarray(np.uint8(self._edges * 255), 'L')

        region_numbers = np.unique(self._regions)
        self._last_num_of_superpixels = region_numbers.shape[0]
        # для нумерации с единицы
        self._regions += 1 if np.amin(region_numbers) == 0 else 0

        self._rgb_marked_image = Image.fromarray(
            np.uint8(mark_boundaries(self._rgb_input_image_ar, self._regions) * 255))

        print(f"numeration starts {np.min(np.unique(self._regions))} and ends {np.max(np.unique(self._regions))}")
        print(f"superpixels num is {len(np.unique(self._regions))}")
        return

    # закрашивает суперпиксели в соответсвии с данными метками и чувствительностью
    def draw_regions(self, filled_mask: np.ndarray, marker_mask: npt.NDArray[np.bool] = None,
                     curr_marker: str = "None", sens: float = 0, change_segmenter_mask=True,
                     save_state=True) -> np.ndarray:
        """
        :param filled_mask: ndarray(image.height, image.width) - массив заполненный метками маркеров
        :param marker_mask: ndarray(image.height, image.width) - массив с картой последнего штриха пользователя
        :param curr_marker:  str - имя маркера, которым пользователь только что закончил рисовать
        :param sens: float - чувствительность
        :param change_segmenter_mask: нужно ли менять маску сегментатора на ту, что нарисует этот метод
        :param save_state: нужно ли сохранять предыдущее состояние
        :return метод возвращает результирующую маску

        draw_refions(mask) - просто отрисует маску с маркерами заданными при создании сегментатора
        """

        if curr_marker == "None" or marker_mask is None:
            mask_rgb = self._get_rgb_mask(mask_template=filled_mask)
            self._rgb_marked_image = self._make_image(mask_rgb)
            if change_segmenter_mask:
                self._mask = filled_mask.copy()
            return self._mask.copy()

        if filled_mask.shape != marker_mask.shape:
            raise ValueError("filled mask and marker mask must have the same shape")

        res_mask = np.copy(filled_mask)
        curr_marker_idx = self._markers[curr_marker][0]

        print('getting marked regions')
        # номера суперпикселей, на которые попал маркер
        marked_regions_nums = self._get_marked_regions(marker_mask)
        print(len(marked_regions_nums))

        print('getting regions to reassign')
        other_markers_mask = (filled_mask != 0) & (filled_mask != curr_marker_idx)
        nums_of_regions_to_reassign = np.unique(self._regions[marker_mask & other_markers_mask])
        print(len(nums_of_regions_to_reassign))

        do_additional_segmentation = False if nums_of_regions_to_reassign.shape[0] == 0 else True

        if do_additional_segmentation:
            print('DOING ADDITIONAL SEGMENTATION')
            self._do_segmentation(nums_of_regions_to_reassign, marker_mask, res_mask)
            marked_regions_nums = self._get_marked_regions(marker_mask)

        print('processing area')
        processed_area = []
        for region_num in marked_regions_nums:
            #  окно просморта -> номера суперпикселей, попавших в окно
            radius = int(self._thresholds['radius'] * sens)
            print(f"radius = {radius}")
            superpixels_around = self._superpixels_around(region_num, radius)
            print(f"sps araound = {superpixels_around}")
            properties = self._region_property(region_num)

            # не затирать уже отмеченные
            for superpixel_num in superpixels_around:
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
                    # mask_rgb[self._regions == superpixel_num, ...] = self._colours_rgb[curr_marker_idx]

            res_mask[self._regions == region_num] = curr_marker_idx  # отметили суперпиксель

        if save_state:
            # marker mask with current marker index instead bool
            m = np.where(marker_mask, curr_marker_idx, 0)
            m = m.astype(dtype='uint8')
            self.push_state(filled_mask, m, curr_marker)

        if change_segmenter_mask:
            self._mask = res_mask.copy()
        mask_rgb = self._get_rgb_mask(mask_template=res_mask)
        self._rgb_marked_image = self._make_image(mask_rgb)
        return res_mask

    # перерисовывает маску с новым значением чувствительности
    def new_sens(self, val: float) -> np.array:
        mask, marker_mask, marker, _ = self._get_state(idx=-1)
        print(
            f"mask = {type(mask)}, marker_mask = {type(marker_mask)}, marker = {type(marker)}, pixels = {len(np.unique(_))}")
        self._mask = self.draw_regions(
            mask, marker_mask != 0, marker, val, change_segmenter_mask=False, save_state=False
            # все равно вернется маска
        )
        return self._mask.copy()

    def push_state(self, mask: np.array, marker_mask: np.array, curr_marker, change_mask=False):
        if change_mask:
            self._mask = mask.copy()

        self._user_marks += marker_mask
        self._states_stack.append((mask.copy(), marker_mask.copy(), curr_marker, np.copy(self._regions)))
        return

    def pop_state(self):
        """
        меняет текущую маску на предыдущую, текущая маска и последние штрихи теряется
        :return:
        """
        if len(self._states_stack) == 0:
            return None
        state = self._states_stack.pop()
        mask = np.copy(state[0])
        marker_mask = state[1]
        self._user_marks -= marker_mask
        self._regions = np.copy(state[-1])
        self.draw_regions(mask, change_segmenter_mask=True, save_state=False)
        return state

    @property
    def mask(self):
        return self._mask.copy()

    @property
    def rgb_marked_image(self):
        return self._rgb_marked_image

    def states_len(self):
        return len(self._states_stack)

    def get_user_marks(self) -> np.array:
        """
        :return: all user marks which WERE SAVED in states stack (draw_refions(save_state=TRUE))
        """
        return np.copy(self._user_marks)

    def _get_state(self, idx: int = -1):
        return self._states_stack[idx]

    def _get_marked_regions(self, marker_mask: npt.NDArray[np.bool]) -> Iterable[int]:
        return np.unique(self._regions[marker_mask])

    def _get_rgb_mask(self, mask_template: np.ndarray):
        """
        :param mask_template: np.ndarray of shape (M, D)
        :return: rgb mask with shape (M, D, 3)
        """
        mask_rgb = np.zeros((*mask_template.shape, 3))
        for marker_index, marker_color in self._colours_rgb.items():
            mask_rgb[mask_template == marker_index] = marker_color
        return mask_rgb

    def _make_crops(self, nums_of_superpixel: Iterable[int]) -> List[Tuple[Point, Point]]:
        # height, width = self._regions.shape
        res = []
        for i in nums_of_superpixel:
            row_indexes, column_indexes = np.where(self._regions == i)

            ymin, ymax = np.min(row_indexes), np.max(row_indexes)
            xmin, xmax = np.min(column_indexes), np.max(column_indexes)

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
            res.append((Point(x=xmin, y=ymin), Point(x=xmax, y=ymax)))
        return res

    def _do_segmentation(
            self, nums_of_regions_to_reassign: Iterable[int],
            mm: npt.NDArray[np.bool],
            filled_mask: np.ndarray):

        crops = self._make_crops(nums_of_regions_to_reassign)

        for crop, i in zip(crops, nums_of_regions_to_reassign):
            up_left, down_right = crop
            row_slice = slice(up_left.y, down_right.y + 1)
            column_slice = slice(up_left.x, down_right.x + 1)
            img_crop = self._rgb_input_image_ar[row_slice, column_slice]

            superpixel_mask = self._regions == i

            previous_marks_in_curr_pixel = np.where(superpixel_mask, self._user_marks, 0)
            m_idx = np.unique(previous_marks_in_curr_pixel)
            print(m_idx)

            if m_idx.shape[0] > 2:
                raise ValueError("Two markers in one superpixel")

            marked_by_user = True if max(m_idx) != 0 else False  # если нет штрихов,то пиксель закрасил метод, а не user

            filled_mask[superpixel_mask] = 0

            # todo придумать что делать если пересекаются два штриха разных маркеров

            # todo придумать что лучше делать с закрашеным пикселем (разбиваются пока маркеры не в разных пикселях)
            # todo как-то учитывать размер кропа для количества суперпикселей (решается циклом)
            # height = down_right.y - up_left.y + 1
            # width = down_right.x - up_left.x + 1

            # нумерация с 0 или 1 неважно
            good_segmentation = False
            labels_for_segmentation = 5
            new_regions = None

            while not good_segmentation:
                print('do segmentation')
                labels_for_segmentation *= 2
                new_regions = slic(img_crop, n_segments=labels_for_segmentation, compactness=5, sigma=1, start_label=1)
                new_regions = np.where(superpixel_mask[row_slice, column_slice], new_regions, 0)

                if not marked_by_user:
                    print('marked by method')
                    good_segmentation = True
                else:
                    print('marked_by_user')

                    new_pixels_under_old_marks = np.unique(np.where(
                        previous_marks_in_curr_pixel[row_slice, column_slice] != 0, new_regions, 0
                    ))
                    new_pixels_under_old_marks = new_pixels_under_old_marks[new_pixels_under_old_marks != 0]
                    print(f"new pixels under OLD marks {new_pixels_under_old_marks}")

                    mm_in_superpixel = np.where(superpixel_mask, mm, 0)
                    new_pixels_under_new_marks = np.unique(np.where(
                        mm_in_superpixel[row_slice, column_slice] != 0, new_regions, 0
                    ))
                    new_pixels_under_new_marks = new_pixels_under_new_marks[new_pixels_under_new_marks != 0]
                    print(f"new pixels under NEW marks {new_pixels_under_new_marks}")

                    if np.intersect1d(
                            new_pixels_under_old_marks,
                            new_pixels_under_new_marks,
                            assume_unique=True
                    ).shape[0] == 0:
                        good_segmentation = True

            # перенумерация с нуля
            new_superpixel_numbers_in_old = np.unique(new_regions)
            for j, num in enumerate(new_superpixel_numbers_in_old, start=0):
                new_regions[new_regions == num] = j

            new_regions += 1 + self._last_num_of_superpixels

            print(
                f'num of new regions={np.unique(new_regions).shape[0]} must be == {new_superpixel_numbers_in_old.shape[0]}')

            self._regions[row_slice, column_slice] = np.where(
                superpixel_mask[row_slice, column_slice], new_regions, self._regions[row_slice, column_slice])
            self._last_num_of_superpixels += new_superpixel_numbers_in_old.shape[0]

            # закраска новых пикселей старыми маркерами
            if marked_by_user:
                new_pixels_under_old_marks = np.unique(np.where(
                    previous_marks_in_curr_pixel != 0, self._regions, 0
                ))
                for j in new_pixels_under_old_marks:
                    filled_mask[self._regions == j] = max(m_idx)
        return

    # возвращет номера суперпикселей вокруг
    # TODO поменять этот метод, чтобы возраващал ток соседние суперпиксели, а не шарился по всему изображеию
    def _superpixels_around(self, region_num: int, radius: int):
        if radius == 0:
            return []
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

    def _make_image(self, mask_rgb: np.ndarray, alpha=0.5):
        img_ar = np.where(
            mask_rgb,
            self._gray_ar[:, :, np.newaxis] * alpha + (1 - alpha) * mask_rgb,
            self._rgb_input_image_ar
        )

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
