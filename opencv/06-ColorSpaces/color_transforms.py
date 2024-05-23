import sys
from pathlib import Path

import cv2
import numpy as np

from ss_util import AssetType
from ss_util import GetAssetPath
from ss_color import CreateImageFromHistogram
from ss_color import GenerateHistogram
from ss_showimages import ShowImages, ImageList
from ss_showimages import TextImage
from ss_showimages import CreateShowCalculatingProcess

def main(argc: int, argv: list[str]) -> int:
    imageList: ImageList = ImageList(1024, 4, 20)
    windowName = "Window"

    filePath = GetAssetPath("girl.jpg", AssetType.Image, Path(sys.argv[0]).parent)
    originalImage = cv2.imread(filePath.as_posix())
    imageList.append(TextImage("Original", originalImage))

    calculatingWidth: int = 500
    calculatingHeight: int = 200
         
    calculatingProcess = CreateShowCalculatingProcess(windowName, calculatingWidth, calculatingHeight)
    calculatingProcess.start()

    histogram = GenerateHistogram(originalImage)
    redChannelImage = CreateImageFromHistogram(histogram[0], originalImage.shape[1], originalImage.shape[0], [0, 0, 255])
    imageList.append("Red Channel", redChannelImage)
    greenChannelImage = CreateImageFromHistogram(histogram[1], originalImage.shape[1], originalImage.shape[0], [0, 255, 0])
    imageList.append("Green Channel", greenChannelImage)
    blueChannelImage = CreateImageFromHistogram(histogram[2], originalImage.shape[1], originalImage.shape[0], [255, 0, 0])
    imageList.append("Blue Channel", blueChannelImage)

    hsvImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
    equalizedHSVImage = hsvImage.copy()
    equalizedHSVImage[:,:,2] = cv2.equalizeHist(hsvImage[:,:,2])
    imageList.append("Equalized HSV", cv2.cvtColor(equalizedHSVImage, cv2.COLOR_HSV2BGR))

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    claheImage = hsvImage.copy()
    claheImage[:,:,2] = clahe.apply(claheImage[:,:,2])
    imageList.append("CLAHE", cv2.cvtColor(claheImage, cv2.COLOR_HSV2BGR))

    toneAdjustedImage = originalImage.copy()
    originalCurve = np.array([0, 50, 100, 150, 200, 255])
    redCurve =      np.array([0, 40, 80, 120, 180, 255])
    blueCurve =     np.array([0, 60, 120, 160, 220, 255])
    fullRange: np.ndarray = np.arange(0, 256)
    redLookUpTable = np.interp(fullRange, originalCurve, redCurve)
    blueLookUpTable = np.interp(fullRange, originalCurve, blueCurve)
    toneAdjustedImage[:,:,2] = cv2.LUT(toneAdjustedImage[:,:,2], redLookUpTable)
    toneAdjustedImage[:,:,0] = cv2.LUT(toneAdjustedImage[:,:,0], blueLookUpTable)
    imageList.append("Tone Adjusted", toneAdjustedImage)

    calculatingProcess.kill()
    ShowImages(windowName, imageList)

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))
