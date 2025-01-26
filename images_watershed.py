import argparse
import sys
from skimage import filters
from concurrent import futures
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

from os import path
import grpc
import numpy as np
from PIL import Image

from skimage.segmentation import mark_boundaries
from skimage.segmentation import watershed
from skimage.util import img_as_ubyte, img_as_float

import matplotlib.image as mp_img


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segmenter_service", description="Run segmenter service"
    )
    parser.add_argument(
        "-p",
        "--path",
        help="path to image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--n_segments",
        help="n of segments",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "-c",
        "--compactness",
        help="compactness",
        type=float,
        default=0,
        required=False,
    )
    
    return parser



def save_image(arr: np.ndarray, name: str, dir_name: str, format: str):
    byte_img = img_as_ubyte(arr)
    img = Image.fromarray(byte_img, format)
    img.save(path.join(dir_name, name))


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    dir_name = path.dirname(args.path)
    

    source_image = Image.open(args.path).convert("RGB")

    gray_image = source_image.convert("L")
    gray_image.save(path.join(dir_name, "gray.png"))
    
    gray = np.asarray(gray_image)
    
    edges = filters.sobel(gray)
    save_image(edges, "gradient.png", dir_name=dir_name, format="L")
    
    regions = watershed(
                edges,
                markers=args.n_segments,
                compactness=args.compactness,
            )  # 0.001
    
    
    image_with_regions = img_as_ubyte(mark_boundaries(img_as_float(source_image), regions))
    save_image(image_with_regions, "regions.png", dir_name=dir_name, format="RGB")
    
    
    kek = cv2.resize(edges, (100,100))
    xx, yy = np.mgrid[0:kek.shape[0], 0:kek.shape[1]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.plot_surface(xx, yy, kek ,rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=0)
    plt.show()
    
   