import sys
from pathlib import Path
import cv2
import numpy as np

maxScaleUp = 100
scalePercent = 0
scaleType = 0
maxType = 1

windowName = "Resize Image"
trackbarValue = "Scale"
trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"
filename = "truth.png"

cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

def GetImagePath(filename) -> Path:
    return Path(sys.argv[0]).parent / filename

originalImage = cv2.imread(GetImagePath(filename).as_posix())
if originalImage is None:
    raise FileNotFoundError("Unable to load file " + filename)

def ShowScaledImage():
    global originalImage
    global scalePercent
    global scaleType

    scaleFactor = 1 + scalePercent / 100.0 if scaleType == 0 else np.maximum(0.2, 1 - scalePercent / 100.0)

    scaledImage = cv2.resize(
        originalImage, None, 
        fx=scaleFactor,
        fy=scaleFactor,
        interpolation=cv2.INTER_LINEAR)
    
    finalImage = originalImage.copy()
    def x():
    #if scaleFactor < 1:
        finalWidth, finalHeight = finalImage.shape[0:2]
        cv2.rectangle(finalImage, (0, 0), (finalWidth - 1, finalWidth - 1), (30, 30, 30), -1)
        scaledHeight, scaledWidth = scaledImage.shape[0:2]
        finalImage[
            int(finalHeight / 2 - scaledHeight / 2):int(finalHeight / 2 + scaledHeight / 2), 
            int(finalWidth / 2 - scaledWidth / 2):int(finalWidth / 2 + scaledWidth / 2)
            ] = scaledImage[:,:]
        cv2.imshow(windowName, finalImage)
    #else:
    #    cv2.imshow(windowName, scaledImage)
    cv2.imshow(windowName, scaledImage)

def ScaleImage(*args):
    global scaleType
    global scalePercent

    scalePercent = args[0]
    
    ShowScaledImage()

def ScaleTypeImage(*args):
    global scaleType
    global scalePercent

    scaleType = args[0]
    scalePercent = 1

    cv2.setTrackbarPos(trackbarValue, windowName, 0)

def main(argc: int, argv) -> int:
    cv2.createTrackbar(trackbarValue, windowName, scalePercent, maxScaleUp, ScaleImage)
    cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, ScaleTypeImage)

    ShowScaledImage()

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))