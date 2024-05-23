import sys

import cv2
import numpy as np

from ss_util import GetAssetPath
from ss_util import AssetType
from ss_showimages import TextImage
from ss_showimages import ImageList
from ss_showimages import ShowImages

def main(argc: int, argv: list[str]) -> int:
    imageList: ImageList = ImageList(1300, 2, 32)

    imageBackground: np.ndarray = cv2.imread(GetAssetPath("sky.jpg", AssetType.Image).as_posix())
    assert imageBackground is not None, "Unable to load image."
    imageList.append("Sky", imageBackground)
    print(f"{imageBackground.shape = }")

    imagePlane: np.ndarray = cv2.imread(GetAssetPath("airplane.jpg", AssetType.Image).as_posix())
    assert imagePlane is not None, "Unable to load image."
    imageList.append("Plane", imagePlane)
    print(f"{imagePlane.shape = }")

    ShowImages("Cloning", imageList)


    return 0



if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))


# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (800,100)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)