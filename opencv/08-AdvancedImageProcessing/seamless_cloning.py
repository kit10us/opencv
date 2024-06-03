import sys
from typing import Sequence

import cv2
import numpy as np

from ss_util import GetAssetPath
from ss_util import AssetType
from ss_showimages import TextImage
from ss_showimages import ImageList
from ss_showimages import ShowImages


def main(argc: int, argv: list[str]) -> int:
    imageList: ImageList = ImageList(800, 2, 40)

    imageBackground: np.ndarray = cv2.imread(GetAssetPath("sky.jpg", AssetType.Image).as_posix())
    assert imageBackground is not None, "Unable to load image."
    imageList.append("Sky", imageBackground)

    imagePlane: np.ndarray = cv2.imread(GetAssetPath("airplane.jpg", AssetType.Image).as_posix())
    assert imagePlane is not None, "Unable to load image."
    imageList.append("Plane", imagePlane)

    # Create a rough mask around the airplane.
    src_mask = np.zeros(imageBackground.shape, imageBackground.dtype)
    poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
    mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    center: list[int] = [800, 100]
    imageCloned = cv2.seamlessClone(imagePlane, imageBackground, mask, center, cv2.NORMAL_CLONE)
    cv2.circle(imageCloned, center, 10, [255, 0, 0], 3)
    
    imageList.append("Cloned", imageCloned)

    ShowImages("Cloning", imageList)

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))

