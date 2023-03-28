import argparse
import sys
from concurrent import futures
from icecream import ic

import grpc
import numpy as np
from PIL import Image

from segmenter import Segmenter
import segmenter_service_pb2 as pb2, segmenter_service_pb2_grpc as pb2_grpc

import utils


class SegmenterService(pb2_grpc.SegmenterServicer):
    def __init__(self):
        self.segmenter = None

    def loadImage(self, request: pb2.LoadImageRequest, context):
        """
        Создает сегментатор для изображения, с пронумерованным списком маркеров

        Args:
            request.markers (_type_): Dict[str, Tuple(int, str)
            request.method_params.method (_type_): Literal["slic", "watershed", "quick_shift", "fwb"]
            request.method_params.params (_type_): Dict[str, float]
        """
        img = None

        try:
            img = Image.open(request.path).convert("RGB")
        except:
            ar = utils.array_from_NdArray(request.image_array)
            img = Image.fromarray(ar)

        method, params = request.method_params.method_name, request.method_params.params
        self.segmenter = Segmenter(
            image=img, markers=request.markers, method=method, **params
        )
        return pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def sendMarkerMask(self, request: pb2.MarkerMask, context):
        """
        Отправляет метки пользователя в виде маски
        
        Args:
            request (_type_): MarkerMask
        """

        marker_mask = utils.array_from_Mask2D(request.mask)

        try:
            self.segmenter.mark_regions(
                marker_mask=marker_mask != 0,
                curr_marker=request.marker,
                sens=request.sens.value,
                change_segmenter_mask=True,
                save_state=True,
            )
        except Exception as e:
            print(e)

        return pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    # def MakeSegmenter(self, image: Image, markers: Dict[str, Tuple(int, str)],
    #              method: Literal["slic", "watershed", "quick_shift", "fwb"],
    #              **method_params):
    #     self.segmenter = Segmenter(image, markers, method, **method_params)

    # def GetMask(self, marker_mask: npt.NDArray, curr_marker: str = "None", sens: float = 0) -> np.ndarray:
    #     new_mask = self.segmenter.mark_regions(
    #         self.segmenter.mask,
    #         marker_mask != 0,  # must be bool
    #         curr_marker,
    #         sens,
    #         change_segmenter_mask=True,
    #         save_state=True
    #     )
    #     return new_mask

    def sendNewSens(self, request: pb2.SensitivityValue, context):
        self.segmenter.new_sens(sens_val=request.value)
        return pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def newMethod(self, request: pb2.NewMethodRequest, context):
        self.segmenter.new_method(
            method=request.method_params.method_name,
            save_marked_regions=request.save_current_mask,
            **request.method_params.params,
        )
        return pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def getMarkedMask2D(self, request, context) -> pb2.Mask2D:
        mask = self.segmenter.mask
        return SegmenterService._Mask2D_from_array(mask)

    def getRegions(self, request, context) -> pb2.Mask2D:
        regions = self.segmenter.regions.astype(dtype=np.int32)
        return SegmenterService._Mask2D_from_array(regions)

    def popState(self, request, context) -> pb2.Mask2D:
        if self.segmenter.states_len() == 0:
            return SegmenterService._empty_Mask2D()
        old_mask = self.segmenter.pop_state()
        return SegmenterService._Mask2D_from_array(old_mask)

    @staticmethod
    def _Mask2D_from_array(arr: np.ndarray) -> pb2.Mask2D:
        height, width = arr.shape
        data_type = str(arr.dtype)
        data = arr.tobytes()
        return pb2.Mask2D(data=data, height=height, width=width, dtype=data_type)

    @staticmethod
    def _empty_Mask2D() -> pb2.Mask2D:
        return pb2.Mask2D(data=np.empty((0, 0)).tobytes(), height=0, width=0, dtype="")


def serve(port):
    print(f"Launching segmenter service at port {port}")
    sys.stdout.flush()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

    pb2_grpc.add_SegmenterServicer_to_server(SegmenterService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print("connection is set up")
    server.wait_for_termination()


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segmenter_service", description="Run segmenter service"
    )
    parser.add_argument(
        "-p",
        "--port",
        help="the port that the application listens to",
        type=str,
        required=True,
    )
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    serve(port=args.port)
