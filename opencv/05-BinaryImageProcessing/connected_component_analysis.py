##
## Show Images module
## by Stacy Smith
##

import sys
from pathlib import Path
import cv2
import numpy as np

from ss_showimages import ShowImages, ImageList, TextImage


def GetImagePath(filename) -> Path:
    localPath: Path = Path(sys.argv[0]).parent / filename
    if localPath.exists():
        return localPath
    
    imageFolderPath: Path = Path(sys.argv[0]).parent.parent / "images" / filename
    if imageFolderPath.exists():
        return imageFolderPath
    
    raise FileNotFoundError(f"Could not locate file {filename}")


def PerformCCA(sourceImage: cv2.Mat):
    # Convert image to grey scale
    greyImage: cv2.Mat = None
    try:
        greyImage: cv2.Mat = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    except:
        greyImage: cv2.Mat = sourceImage
        pass

    # Convert to binary
    _, thresholdImage = cv2.threshold(greyImage, 50, 255, cv2.THRESH_BINARY)

    # Perform Connected Component Analysis
    imageLabels: cv2.Mat
    _, imageLabels = cv2.connectedComponents(thresholdImage)
    minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(imageLabels)
    imageLabels = 255 * (imageLabels - minValue) / (maxValue - minValue)
    imageLabels = np.uint8(imageLabels)
    imageColorMap: cv2.Mat = cv2.applyColorMap(imageLabels, cv2.COLORMAP_JET)
    
    windowName = "Window"
    ShowImages(windowName, \
        ImageList(1024, 3, 20, \
            [TextImage("Grey", greyImage), TextImage("Threshold", thresholdImage), TextImage(f"CCA ({imageLabels.max()})", imageColorMap)]) \
    )


def main(argc: int, argv: list[str]) -> int:
    amoebaImagePath = GetImagePath("7.png")

    amoebaImage = cv2.imread(amoebaImagePath.as_posix())
    if amoebaImage is None:
        raise Exception(f"Failed to load image {amoebaImagePath.as_posix()}")
                        
    PerformCCA(amoebaImage)

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))