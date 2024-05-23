import sys
from pathlib import Path
import cv2
import math
import numpy as np

from ss_showimages import ShowImages, ImageList
from ss_showimages import TextImage

def main(argc: int, argv: list[str]) -> int:
    imageRGBRed = np.zeros([256, 256, 3])
    imageRGBGreen = np.zeros([256, 256, 3])
    imageRGBBlue = np.zeros([256, 256, 3])
    for y in range(0, 255, 5):
        for x in range(0, 255, 5):
            imageRGBRed[y:y + 5, x:x + 5] = (y, x, 255)
            imageRGBGreen[y:y + 5, x:x + 5] = (y, 255, x)
            imageRGBBlue[y:y + 5, x:x + 5] = (255, y, x)


    red = math.sqrt(255^2 + 255^2)
    print(red)
    
    windowName = "Window"
    ShowImages(windowName, ImageList(1024, 3, 20, [
        TextImage("RGB (Red) Space", imageRGBRed),
        TextImage("RGB (Green) Space", imageRGBGreen),
        TextImage("RGB (Blue) Space", imageRGBBlue)
        ]))
    
    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))