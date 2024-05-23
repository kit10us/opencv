import sys
from pathlib import Path
import cv2

from ss_showimages import ShowImages, TextImage, ImageList
from ss_util import AssetType, GetAssetPath

def DetectCarContour(sourceImage: cv2.Mat):
    # Prepare a copy of the source image for displaying the final results.
    finalImage = sourceImage.copy()

    # 1. Extract the blue channel from the image as it has the relevant information.
    blueChannel = sourceImage[:,:,0]

    # 2. Substract the green channel from the blue channel to get a grey area,
    # the box with the car in it, as well as the outline of the car.
    blueChannelProcessed = blueChannel - sourceImage[:,:,1]
    
    # 3. Convert the grey image to binary, noting that the color will be a low grey.
    _, binaryImage = cv2.threshold(blueChannelProcessed, 30, 255, cv2.THRESH_BINARY)

    # 4. Find the contours of the box around the car.
    contoursBox, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursBoxImage = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contoursBoxImage, contoursBox, 0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.drawContours(finalImage, contoursBox, 0, (0, 255, 0), 3, cv2.LINE_AA)

    # 5. Invert the image so that the car outline is white, for finding contour.
    invertedImage = cv2.bitwise_not(binaryImage)

    # 6. Find the car outline's contour.
    contoursCar, hierarchy = cv2.findContours(invertedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursCarImage = cv2.cvtColor(invertedImage, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contoursCarImage, contoursCar, 0, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.fillPoly(finalImage, pts=[contoursCar[0]], color=(0,0,0))
    cv2.drawContours(finalImage, contoursCar, 0, (0, 255, 255), 3, cv2.LINE_AA)
    
    windowName = "Window"
    ShowImages(windowName, ImageList(1040, 3, 30, [
        TextImage("Original", sourceImage),
        TextImage("Blue Channel", blueChannel),
        TextImage("Blue - Green Processed", blueChannelProcessed),
        TextImage("Binary", binaryImage),
        TextImage("Box Contour", contoursBoxImage),
        TextImage("Inverted", invertedImage),
        TextImage("Car Contour", contoursCarImage),
        TextImage(f"Final", finalImage)
        ]))
    

def main(argc: int, argv: list[str]) -> int:
    sourcePath: Path = GetAssetPath("Quiz-1-Assets.png", AssetType.Image)
    
    sourceImage: cv2.Mat = cv2.imread(sourcePath.as_posix())
    if sourceImage is None:
        raise Exception(f"Unable to read image {sourcePath.as_posix()}")
    
    DetectCarContour(sourceImage)
    

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))
