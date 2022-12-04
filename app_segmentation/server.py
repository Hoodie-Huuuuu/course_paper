from PIL import Image
from segmenter import Segmenter
from typing import *
import numpy as np
import numpy.typing as npt


class Server:
    def __init__(self):
        self.segmenter = None

    def MakeSegmenter(self, image: Image, markers: Dict[str, str],
                 method: Literal["slic", "watershed", "quick_shift", "fwb"],
                 **method_params):
        self.segmenter = Segmenter(image, markers, method, **method_params)

    def GetMask(self, marker_mask: npt.NDArray, curr_marker: str = "None", sens: float = 0) -> np.ndarray:
        new_mask = self.segmenter.mark_regions(
            self.segmenter.mask,
            marker_mask != 0,  # must be bool
            curr_marker,
            sens,
            change_segmenter_mask=True,
            save_state=True
        )
        return new_mask

    def NewSens(self, sens_val: float) -> np.array:
        mask = self.segmenter.new_sens(sens_val=sens_val)
        return mask

    def StatesLen(self) -> int:
        return self.segmenter.states_len()

    def PopState(self):
        if self.segmenter.states_len() == 0:
            # todo поменять для grpc
            return None
        old_mask = self.segmenter.pop_state()
        return old_mask

    # todo Допелить метод
    # def NewMethod(self, method: Literal["slic", "watershed", "quick_shift", "fwb"], save_curr_pixels=False, **method_params):
    #     states = self.segmenter.get_states()
    #     self.segmenter = Segmenter(self._curr_image, self.markers, self.curr_method, **self.params)
    #
    #     self.segmenter.load_user_marks(states, self.sens_val_scale)

    #для отладки - отрисвока границ
    # def GetRegions(self):
    #     return self.segmenter.
