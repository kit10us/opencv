import sys
from pathlib import Path
import cv2
import numpy as np
from enum import Enum
from ss_showimages import ShowImages
from ss_showimages import TextImage

def GetImagePath(filename) -> Path:
    localPath: Path = Path(sys.argv[0]).parent / filename
    if localPath.exists():
        return localPath
    imageDirectoryPath: Path = Path(sys.argv[0]).parent.parent / "images" / filename
    if imageDirectoryPath.exists():
        return imageDirectoryPath
    raise FileNotFoundError(f"File not found {filename}")


def convertBGRtoGray(image: cv2.Mat) -> cv2.Mat:
    redChannel = image[:,:,2]
    greenChannel = image[:,:,1]
    blueChannel = image[:,:,0]
    outputImage = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            gray = redChannel[y, x] * 0.299 + greenChannel[y, x] * 0.587 + blueChannel[y, x] * 0.114
            outputImage[y, x] = np.round(gray)
    return outputImage

def convertBGRtoHSV(image):
    outputImage = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            R = image[y, x, 2] / 255.0
            G = image[y, x, 1] / 255.0
            B = image[y, x, 0] / 255.0
 
            minRGB = min(R, G, B)

            V = max(R, G, B)
            
            if V == minRGB:
                H = 1
                S = 1
            else:
                S = 0 if V == 0 else (V - minRGB) / V
                H: float = 0
                if V == R:
                    H = 60.0 * (G - B) / (V - minRGB)
                elif V == G:
                    H = 120.0 + 60.0 * (B - R) / (V - minRGB)
                elif V == B:
                    H = 240.0 + 60.0 * (R - G) / (V - minRGB)
                else:
                    assert False

                if H < 0:
                    H = H + 360.0
                    
            outputImage[y, x, 0] = int(np.round(H * 0.5))
            outputImage[y, x, 1] = int(np.round(S * 255.0))
            outputImage[y, x, 2] = int(np.round(V * 255.0))
    return outputImage


def main(argc: int, argv: list[str]) -> int:
    imageList: list[TextImage] = []

    # Load original image
    imagePath = GetImagePath("girl.jpg")
    originalImage = cv2.imread(imagePath.as_posix())
    if originalImage is None:
        raise Exception(f"Failed to read image {imagePath.as_posix()}")
    imageList.append(TextImage("Original", originalImage))

    # Convert image to grayscale
    grayscaleImage = convertBGRtoGray(originalImage)
    imageList.append(TextImage("Gray Scale", grayscaleImage))

    # Convert image to grayscale
    grayscaleCV2Image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    imageList.append(TextImage("Gray Scale CV2", grayscaleCV2Image))

    # Convert image to grayscale
    hsvImage = convertBGRtoHSV(originalImage)
    imageList.append(TextImage("HSV Scale", hsvImage))

    # Convert image to grayscale
    hsvCV2Image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
    imageList.append(TextImage("HSV Scale CV2", hsvCV2Image))


    
    windowName = "Window"
    ShowImages(windowName, 1024, 3, 20, imageList)

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))