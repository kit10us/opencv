import sys

import numpy as np
import cv2

from ss_util import AssetType
from ss_util import GetAssetPath
from ss_showimages import TextImage
from ss_showimages import ShowImages, ImageList

def main(argc: int, argv: list[str]) -> int:
    textImages: ImageList = ImageList( 1400, 4, 30)

    # Step 1: Capture images and times
    images: list[np.ndarray] = []
    images.append(cv2.imread(GetAssetPath("img_0.033.jpg", AssetType.Image).as_posix()))
    images.append(cv2.imread(GetAssetPath("img_0.25.jpg", AssetType.Image).as_posix()))
    images.append(cv2.imread(GetAssetPath("img_2.5.jpg", AssetType.Image).as_posix()))
    images.append(cv2.imread(GetAssetPath("img_15.jpg", AssetType.Image).as_posix()))

    textImages.append(TextImage("Timed 0.033", images[0]))
    textImages.append(TextImage("Timed 0.25", images[1]))
    textImages.append(TextImage("Timed 2.5", images[2]))
    textImages.append(TextImage("Timed 15.0", images[3]))

    times: np.ndarray = np.array([0.033, 0.25, 2.5, 15.0], dtype = np.float32)

    # Step 2: Align images
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images) # Doesn't require times

    textImages.append(TextImage("Aligned MTB (0.033)", images[0]))
    textImages.append(TextImage("Aligned MTB (0.25)", images[1]))
    textImages.append(TextImage("Aligned MTB (2.5)", images[2]))
    textImages.append(TextImage("Aligned MTB (15.0)", images[3]))

    # Step 3: Recover camera response function
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)

    textImages.append(TextImage("Calibrated (0.033)", images[0]))
    textImages.append(TextImage("Calibrated (0.25)", images[1]))
    textImages.append(TextImage("Calibrated (2.5)", images[2]))
    textImages.append(TextImage("Calibrated (15.0)", images[3]))

    # Step 4: Merge images
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

    textImages.append(TextImage("Merged", hdrDebevec))

    # Step 5: Tone mapping
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago *= 3

    textImages.append(TextImage("Drago tone mapped", ldrDrago * 255))

    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)

    textImages.append(TextImage("Reinhard tone mapped", ldrReinhard * 255))

    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)

    textImages.append(TextImage("Matiuk tone mapped", ldrMantiuk * 255))

    ShowImages("HDR", textImages)

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))