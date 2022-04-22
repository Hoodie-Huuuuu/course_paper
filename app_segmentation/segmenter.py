import numpy as np
import numpy.typing as npt
from skimage import filters, color
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from PIL import Image, ImageColor
from typing import *
import cv2 as cv
import matplotlib.pyplot as plt


class Segmenter:
    def __init__(self, image: Image, markers: Dict[str, str]):
        """
        :param image: RGB PIL.Image
        :param markers: OrderedDict["marker_name", "hex_color"]
        """

        # пароги для чувствительности
        self.thresholds = {"mean": 20,
                           "var": 20,
                           # в пикселях
                           "radius": 100}
        # маркеры
        self.markers = markers
        # изображение в LAB координатах
        self.lab_image = cv.cvtColor(src=np.asarray(image), code=cv.COLOR_RGB2LAB)
        # преобразование из hex в rgb
        self.hex2rgb = lambda hex_colour: ImageColor.getcolor(hex_colour, "RGB")
        # список цветов в rgb
        self.colours_rgb = {i: self.hex2rgb(hex_color) for i, hex_color in self.markers.values()}

        # входное изображение (изначально без разметок)
        self.rgb_marked_image = image
        self.rgb_input_image_ar = np.asarray(image)
        self.gray_ar = np.asarray(image.convert('L'))

        # метод сегментации (watershed)
        self.edges = filters.sobel(self.gray_ar)
        self.regions = watershed(self.edges, markers=1000, compactness=0.0005)
        return


    # закрашивает суперпиксели в соответсвии с данными метками и чувствительностью
    def draw_regions(self, filled_mask: np.ndarray, marker_mask: np.ndarray = None,
                     curr_marker: str = "None", sens: float = 0):
        """
        :param filled_mask: ndarray(image.height, image.width) - массив заполненный метками маркеров
        :param marker_mask: ndarray(image.height, image.width) - массив с последним штрихом пользователя
        :param curr_marker:  str - имя маркера, которым пользователь только что закончил рисовать
        :param sens: float - чувствительность
        """



        # та жа самая маска, но с цветами, для отображения
        mask_rgb = np.zeros((*filled_mask.shape, 3))
        for marker_index, marker_color in self.colours_rgb.items():
            mask_rgb[filled_mask == marker_index] = self.colours_rgb[marker_index]

        if curr_marker == "None":
            self.rgb_marked_image = self.make_image(mask_rgb)
            return

        if filled_mask.shape != marker_mask.shape:
            raise ValueError("filled mask and marker mask must have the same shape")

        # результат
        res_mask = np.copy(filled_mask)
        curr_marker_idx = self.markers[curr_marker][0]
        # номера суперпикселей, на которые попал маркер
        marked_regions = np.unique(self.regions[marker_mask == curr_marker_idx])

        # для каждого суперпикселя, на который попал последний штрих
        # в радиусе сравниваем характеристики в соотвествии с чувствительностью
        processed_area = []
        for region_num in marked_regions:
            #  окно просморта -> номера суперпикселей, попавших в окно
            radius = int(self.thresholds['radius'] * sens)
            sps_around = self.sps_around(region_num, radius)
            properties = self.region_property(region_num)

            for superpixel_num in sps_around:
                if superpixel_num in processed_area:
                    break
                properties_superpixel = self.region_property(superpixel_num)

                flag = True
                for prop_name, prop_val in properties_superpixel.items():
                    if np.linalg.norm(properties[prop_name] - prop_val) > self.thresholds[prop_name] * sens:
                        flag = False
                        break

                if flag:
                    processed_area += superpixel_num
                    res_mask[self.regions == superpixel_num] = curr_marker_idx  # отметили суперпиксель
                    mask_rgb[self.regions == superpixel_num, ...] = self.colours_rgb[curr_marker_idx]


            res_mask[self.regions == region_num] = curr_marker_idx  # отметили суперпиксель
            mask_rgb[self.regions == region_num, ...] = self.colours_rgb[curr_marker_idx]

        # полупрозрачная маска на картинке
        self.rgb_marked_image = self.make_image(mask_rgb)
        return res_mask


    def sps_around(self, region_num: int, radius: int):
        indexes = (np.where(self.regions == region_num))
        length = indexes[0].shape[0]
        indexes = list(zip(indexes[0], indexes[1]))
        center_idx = indexes[length // 2]
        y, x = center_idx

        height, width = self.regions.shape
        x1, y1, x2, y2 = x - radius, y - radius, x + radius, y + radius
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > width:
            x2 = width - 1
        if y2 > height:
            y2 = height - 1

        return list(np.unique(self.regions[y1: y2+1, x1: x2+1]))


    def make_image(self, mask_rgb: np.ndarray, alpha=0.4):
        img_ar = np.where(mask_rgb, self.gray_ar[:, :, np.newaxis] * alpha, self.rgb_input_image_ar)
        img_ar = img_ar + (1 - alpha) * mask_rgb
        img = Image.fromarray(np.clip(img_ar.astype(int), 0, 255).astype('uint8'), "RGB")
        return img


    def region_property(self, number: int) -> Dict[str, float]:
        mask = (self.regions == number)
        count = np.sum(mask)
        mean = np.sum(self.lab_image[mask, ...], axis=0) / count
        # mean = np.linalg.norm(mean)
        variance = np.sum((self.lab_image[mask, ...] - mean)**2, axis=0) / count
        # variance = np.linalg.norm(variance)
        return {"mean": mean, "var": variance}







