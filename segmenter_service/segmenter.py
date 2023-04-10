from dataclasses import dataclass
from typing import *

import numpy as np
import numpy.typing as npt
from PIL import Image
from icecream import ic
from skimage import filters
from skimage.color import rgb2lab
import bisect


import utils


# TODO иногда возникают исключения если рисовать у края картинки (косяк с канвасом, а не сегментатором)


@dataclass
class Point:
    x: int
    y: int


class Segmenter:
    def __init__(
        self,
        image: Image,
        markers: Dict[str, str],
        method: Literal["slic", "watershed", "quick_shift", "fwb"],
        **method_params,
    ):
        """
        :param image: RGB PIL.Image
        :param markers: OrderedDict["marker_name", "hex_color"]

        Сначала вызывается метод draw
        отрисованное изображение доступно через свойство rgb_marked_image
        """

        # пароги для чувствительности
        self._thresholds = {"mean": 30, "var": 30, "radius": 100}  # в пикселях
        # маркеры
        self._markers = markers
        ic(markers)
        
        self.image_height, self.image_width = image.height, image.width
        

        # стек состояний маски
        # на любом шаге лежит (предыдущая маска, новые штрихи на этой маске, номер маркера, деление на суперпиксели)
        self._states_stack = []

        # выходная маска сегментатора и все штрихи пользователя (будет отправлятся как массив байт сервером)
        self._mask = np.zeros((image.height, image.width), dtype="uint8")

        # TODO отправлять как 1 json file сервером
        self._user_marks = np.zeros((image.height, image.width), dtype="uint8")

        gray = np.asarray(image.convert("L"))
        self._img_repr = {
            "lab": rgb2lab(np.asarray(image)),
            "rgb": np.asarray(image),
            "gray": gray,
            "edges": filters.sobel(gray),
        }

        self._regions, self._last_num_of_superpixels = utils.segmentation(
            image_repr=self._img_repr, method=method, **method_params,
        )

        print(
            f"numeration starts {np.min(np.unique(self._regions))} and ends {np.max(np.unique(self._regions))}"
        )
        print(f"num of superpixels is {len(np.unique(self._regions))}")
        return

    # def save_regions(self, filled_mask: np.ndarray):
    #     if self._mask.shape != filled_mask.shape:
    #         raise ValueError(f"cant save mask, shape must be {self._mask.shape}, but got {filled_mask.shape}")
    #     self._mask = filled_mask.copy()
    #     return

    # закрашивает суперпиксели в соответсвии с данными метками и чувствительностью
    def mark_regions(
        self,
        filled_mask: np.ndarray = None,
        marker_mask: npt.NDArray[np.bool] = None,
        curr_marker: str = "None",
        sens: float = 0,
        change_segmenter_mask=True,
        save_state=True,
    ) -> np.ndarray:
        """
        :param filled_mask: ndarray(image.height, image.width) - массив заполненный метками маркеров
        :param marker_mask: ndarray(image.height, image.width) - булевский массив с картой последнего штриха пользователя
        :param curr_marker: str - имя маркера, которым пользователь только что закончил рисовать
        :param sens: float - чувствительность
        :param change_segmenter_mask: нужно ли менять маску сегментатора на ту, что нарисует этот метод
        :param save_state: нужно ли сохранять предыдущее состояние
        :param alpha: прозрачность цветной маски
        :return метод возвращает результирующую маску

        mark_regions(mask) - просто
        """
        if filled_mask is None:
            filled_mask = self.mask
            
        ic(sens)

        if curr_marker == "None" or marker_mask is None:
            raise ValueError("'curr_marker' and 'marker_mask' are required parameters")

        if filled_mask.shape != marker_mask.shape:
            raise ValueError("filled mask and marker mask must have the same shape")

        res_mask = np.copy(filled_mask)
        curr_marker_idx = self._markers[curr_marker]

        print("\ngetting marked regions")
        # номера суперпикселей, на которые попал маркер
        marked_regions_nums = self._get_marked_regions(marker_mask)
        print(f"got {len(marked_regions_nums)}")

        print("\ngetting regions to reassign")
        other_markers_mask = (filled_mask != 0) & (filled_mask != curr_marker_idx)
        nums_of_regions_to_reassign = np.unique(
            self._regions[marker_mask & other_markers_mask]
        )
        print(f"num of regions to reassign {len(nums_of_regions_to_reassign)}")

        do_additional_segmentation = (
            False if nums_of_regions_to_reassign.shape[0] == 0 else True
        )

        if do_additional_segmentation:
            print("\nDOING ADDITIONAL SEGMENTATION")
            self._do_segmentation(nums_of_regions_to_reassign, marker_mask, res_mask)
            marked_regions_nums = self._get_marked_regions(marker_mask)

        # TODO переделать сканировние окрестности и не перекрашивать закрашенные пиксели
        # print('\nprocessing area')
        
        processed_area = [] # номера обработанных
        
        reference_regions = list(marked_regions_nums) # sorted
        reference_props = [None for i in reference_regions] # same size
        
        if sens > 0:
            # считаем свойства для пикселей
            print("count props")
            reference_props = [
                self._region_property(num)
                for num in reference_regions
            ]
        
        while len(reference_regions) > 0:
            print("while iteration")
            next_reference_regions = [] # пиксели для обработки на след цикле
            next_reference_props = []
            
            print("i go through refereence pixels")
            for i in range(len(reference_regions)):
                num_reference_region, reference_prop = reference_regions[i], reference_props[i]
                
                ic(reference_regions)
                bisect.insort(processed_area, num_reference_region)
                ic(processed_area)
                
                res_mask[
                        self._regions == num_reference_region
                ] = curr_marker_idx
                
                if not sens > 0:
                    print("dont do scan sens = 0")
                    continue
                
                print("do scan")
                    
                empty_superpixels_around = self._empty_superpixels_around(num_reference_region)
                ic(empty_superpixels_around)
                
                for superpixel_num in empty_superpixels_around:
                    print("count props for")
                    ic(superpixel_num)
                    
                    # если уже обработан то продолжить
                    if utils.in_sorted_list(superpixel_num, processed_area):
                        print(f"{superpixel_num} has been already processed")
                        continue
                    
                    if utils.in_sorted_list(superpixel_num, next_reference_regions):
                        print(f"{superpixel_num} already in regions to process")
                        continue
                    
                    # сравниваем соседние суперпиксели по свойствам
                    props = self._region_property(superpixel_num)
                    if utils.properties_equal(props, reference_prop, self._thresholds, sens):
                        print(f"{superpixel_num} хороший")
                        idx = bisect.bisect_left(next_reference_regions, superpixel_num)
                        next_reference_regions.insert(idx, superpixel_num)
                        next_reference_props.insert(idx, props)
                    
                    else:
                        print(f"{superpixel_num} плохой")
                        bisect.insort(processed_area, superpixel_num)
                print("\n\n\nend of scan")
                ic(reference_regions)
                ic(next_reference_regions)
                
            reference_regions = next_reference_regions
            reference_props = next_reference_props
        print("\n\nout of while")     
                    
        if save_state:
            # marker mask with current marker index instead bool
            m = np.where(marker_mask, curr_marker_idx, 0)
            m = m.astype(dtype="uint8")
            self.push_state(filled_mask, m, curr_marker)

        if change_segmenter_mask:
            self._mask = res_mask.copy()

        return res_mask

    def new_sens(self, sens_val: float) -> np.array:
        """
        перерисовывает маску с новым значением чувствительности
        :param sens_val: новое значение чувствительности
        :param transparency: прозрачность
        :return: копия маски
        """
        if self.states_len == 0:
            return
        mask, marker_mask, marker, _ = self._get_state(idx=-1)
        print(
            f"mask = {type(mask)}, marker_mask = {type(marker_mask)}, marker = {type(marker)}, pixels = {len(np.unique(_))}"
        )
        self.mark_regions(
            mask,
            marker_mask != 0,
            marker,
            sens_val,
            change_segmenter_mask=True,
            save_state=False,
            # все равно вернется маска
        )
        return

    # todo Пушить прдыдущее состояние
    def new_method(
        self,
        method: Literal["slic", "watershed", "quick_shift", "fwb"],
        save_marked_regions=True,
        **method_params,
    ):

        # пронумерованы с 1
        new_regions, _ = utils.segmentation(
            image_repr=self._img_repr, method=method, **method_params,
        )
        
        marked_zone = self._mask != 0
        old_marked_regions = np.where(
            marked_zone, self._regions, -1
        )  # может быть заполнена только числом -1

        nums = np.unique(old_marked_regions)
        sorted_nums_in_marked_zone = nums[1:]

        sorted_nums_in_new_regions = np.unique(new_regions)

        # пользователь ничего до этого не нарисовал
        if sorted_nums_in_marked_zone.size == 0:
            ic("nothing drawn")
            self._regions = new_regions
            self._last_num_of_superpixels = sorted_nums_in_new_regions[-1]
            return 

        bias = sorted_nums_in_new_regions[-1]
        old_marked_regions += bias  # теперь старые номера и новые не пересекаются
        sorted_nums_in_marked_zone += bias
        new_regions[marked_zone] = old_marked_regions[marked_zone]

        numbers = np.concatenate(
            (sorted_nums_in_new_regions, sorted_nums_in_marked_zone)
        )
        self._last_num_of_superpixels = utils.renumerate_regions(
            regions=new_regions, sorted_nums_of_pixels=numbers, start=1
        )

        self._regions = new_regions

        return

    # def load_user_marks(self, states, sens: float):
    #     for _, marker_mask, curr_marker, _ in states:
    #         print(curr_marker)
    #         print("marker_mask numbers:", np.unique(marker_mask))
    #         print("mask numbers: ", np.unique(self._mask))
    #         self.draw_regions(
    #             filled_mask=self._mask,
    #             marker_mask=marker_mask != 0,
    #             curr_marker=curr_marker,
    #             sens=sens,
    #             change_segmenter_mask=True,
    #             save_state=True,
    #         )

    def push_state(
        self, mask: np.array, marker_mask: np.array, curr_marker, change_mask=False
    ):
        if change_mask:
            self._mask = mask.copy()

        self._user_marks += marker_mask
        self._states_stack.append(
            (mask.copy(), marker_mask.copy(), curr_marker, np.copy(self._regions))
        )
        return

    def pop_state(self):
        """
        меняет текущую маску на предыдущую, текущая маска и последние штрихи теряется
        :return:
        """
        if len(self._states_stack) == 0:
            return None
        state = self._states_stack.pop()
        self._mask = np.copy(state[0])
        marker_mask = state[1]
        self._user_marks -= marker_mask
        self._regions = np.copy(state[-1])

        return np.copy(self._mask)

    @property
    def mask(self):
        return self._mask.copy()

    @property
    def regions(self):
        return self._regions.copy()

    def get_states(self):
        return self._states_stack

    def states_len(self):
        return len(self._states_stack)

    def get_user_marks(self) -> np.array:
        """
        :return: all user marks which WERE SAVED in states stack (mark_regions(save_state=TRUE))
        """
        return np.copy(self._user_marks)

    def _get_state(self, idx: int = -1):
        return self._states_stack[idx]

    def _get_marked_regions(self, marker_mask: npt.NDArray[np.bool]):  # Iterable[int]
        return np.unique(self._regions[marker_mask])

    def _make_crops(
        self, nums_of_superpixel: Iterable[int]
    ) -> List[Tuple[Tuple[Point, Point], int]]:
        """
        :param nums_of_superpixel: номера суперпикселей, для которых надо сделать кроп
        :return: Возвращает координаты левого верхнего и правого нижнего угла
                 кропа для каждого суперпикселя c его номером - (crop , idx)

        """
        height, width = self._regions.shape
        res = []
        for i in nums_of_superpixel:
            row_indexes, column_indexes = np.where(self._regions == i)

            ymin, ymax = np.min(row_indexes), np.max(row_indexes)
            xmin, xmax = np.min(column_indexes), np.max(column_indexes)
            
            if ymin > 0:
                ymin -= 1
            if xmin > 0:
                xmin -= 1
            if ymax < height - 1:
                ymax += 1
            if xmax < width - 1:
                xmax += 1
                
            res.append(((Point(x=xmin, y=ymin), Point(x=xmax, y=ymax)),i))
        return res

    def _do_segmentation(
        self,
        nums_of_regions_to_reassign: Iterable[int],
        new_marker_mask: npt.NDArray[np.bool],
        filled_mask: np.ndarray,
    ):

        """
        Делает подразбиение пока метки пользователя
        не будут в разных супер пикселях
        и зануляет суперпиксели отмеченые методом анализа соседних областей

        Зануление для того, чтобы метод mark_regions их не подразбивал
        
        :param nums_of_regions_to_reassign: пиксели для подразбиения
        :param new_marker_mask: маска со штрихами
        :param filled_mask: закрашенная маска - ИЗМЕНЯЕТСЯ методом
        :return: nothing
        """

        crops_with_num = self._make_crops(nums_of_regions_to_reassign)

        for crop, i in crops_with_num:
            up_left, down_right = crop
            row_slice = slice(up_left.y, down_right.y + 1)
            column_slice = slice(up_left.x, down_right.x + 1)

            # TODO добавить возможность выбора метода (пока slic)
            img_crop = self._img_repr["rgb"][row_slice, column_slice]

            superpixel_mask = self._regions[row_slice, column_slice] == i

            previous_marks_in_curr_pixel = np.where(
                superpixel_mask, self._user_marks[row_slice, column_slice], 0
            )
            m_idx = np.unique(previous_marks_in_curr_pixel)

            if m_idx.shape[0] > 2:  # 0 from user_marks_mask and idx of marker
                raise ValueError("Two markers in one superpixel")

            # если нет штрихов, то пиксель закрасил метод, а не user
            prev_marker_idx = m_idx[-1] # так как отсортированный то это максимальный элемент
            marked_by_user = True if prev_marker_idx > 0 else False

            if not marked_by_user:
                ic("marked by method")
                filled_mask[row_slice, column_slice][superpixel_mask] = 0
                continue

            # TODO придумать что делать если пересекаются два штриха разных маркеров

            new_regions, n_new_superpixels = utils.additionally_split(
                img_crop=img_crop,
                zone_to_split=superpixel_mask,
                old_marker_mask=previous_marks_in_curr_pixel != 0,
                new_marker_mask=new_marker_mask[row_slice, column_slice],
                n_segments=5,
                numeration_start=1 + self._last_num_of_superpixels,
            )

            # TODO закомментить отладочный вывод
            keks = np.unique(self._regions)
            print(f"OLD N OF REGIONS {keks.size} and max is {keks[-1]}")
            ic(n_new_superpixels)

            self._regions[row_slice, column_slice][superpixel_mask] = new_regions[
                superpixel_mask
            ]
            self._last_num_of_superpixels += n_new_superpixels

            new_superpixels_under_new_marks = np.unique(
                np.where(
                    superpixel_mask & new_marker_mask[row_slice, column_slice],
                    new_regions,
                    -1,
                )
            )

            # TODO закомментить отладочный вывод
            keks_new = np.unique(self._regions)
            print(
                f"NEW N OF REGIONS {keks_new.size} and max is {keks_new[-1]},\nmust be n = {keks.size + n_new_superpixels -1} and max = {keks[-1]+n_new_superpixels}"
            )

            # TODO list = lisr[1:] так как list отстортирован
            new_superpixels_under_new_marks = new_superpixels_under_new_marks[
                new_superpixels_under_new_marks != -1
            ]
            
            # TODO закомментить отладочный вывод
            print(
                f"new pixels under NEW marks in common numeration{new_superpixels_under_new_marks}"
            )

            for num in new_superpixels_under_new_marks:
                new_superpixel_mask = filled_mask[row_slice, column_slice] == num
                filled_mask[row_slice, column_slice][new_superpixel_mask] = 0
        return

    # возвращет номера суперпикселей вокруг
    # TODO поменять этот метод, чтобы возраващал ток соседние суперпиксели, а не шарился по всему изображеию
    def _empty_superpixels_around(self, region_num: int):
        ic(f"looking neighbors for {region_num}")
        region_mask = self.regions == region_num
        idxs = np.where(region_mask)
        
        only_borders = region_mask.copy()
        
        vectors = [-1, 1]
        for i in vectors:
            for j in vectors:
                idxs0 = idxs[0] + i
                idxs1 = idxs[1] + j
                
                idxs_inside_image = np.logical_and.reduce((
                    idxs0 >= 0,
                    idxs0 < self.image_height,
                    idxs1 >= 0,
                    idxs1 < self.image_width
                ))
                
                idxs0 = idxs0[idxs_inside_image]
                idxs1 = idxs1[idxs_inside_image]
                only_borders[idxs0, idxs1] = True
        only_borders[idxs] = False
        neighbors = list(np.unique(self.regions[only_borders]))
        
        crops_with_num = self._make_crops(neighbors)

        for crop, i in crops_with_num:
            up_left, down_right = crop
            row_slice = slice(up_left.y, down_right.y + 1)
            column_slice = slice(up_left.x, down_right.x + 1)

            region_mask = self.regions[row_slice, column_slice] == i
            if np.any(self._mask[row_slice, column_slice][region_mask] != 0):
                del neighbors[bisect.bisect_left(neighbors, i)]
        return neighbors
    
    # TODO сделать получение свойств для кропа, а не сомтреть малюсенький суперпиксель на целой маске
    def _region_property(self, number: int) -> Dict[str, float]:
        mask = self._regions == number
        count = np.sum(mask)
        mean = np.sum(self._img_repr["lab"][mask, ...], axis=0) / count
        # mean = np.linalg.norm(mean)
        variance = (
            np.sum((self._img_repr["lab"][mask, ...] - mean) ** 2, axis=0) / count
        )
        # variance = np.linalg.norm(variance)
        return {"mean": mean, "var": variance}

