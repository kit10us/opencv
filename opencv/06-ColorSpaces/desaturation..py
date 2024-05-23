import sys
from pathlib import Path
import cv2
import numpy as np

from ss_showimages import ShowImages, ImageList
from ss_showimages import TextImage

def GetImagePath(filename) -> Path:
    localPath: Path = Path(sys.argv[0]).parent / filename
    if localPath.exists():
        return localPath
    imageDirectoryPath: Path = Path(sys.argv[0]).parent.parent / "images" / filename
    if imageDirectoryPath.exists():
        return imageDirectoryPath
    raise FileNotFoundError(f"Unable to find file {filename}")

def main(argc: int, argv: list[str]) -> int:
    imageList: ImageList = ImageList(940, 3, 20)

    # Load original image
    imagePath = GetImagePath("girl.jpg")
    originalImage = cv2.imread(imagePath.as_posix())
    if originalImage is None:
        raise Exception(f"Unable to read file {imagePath.as_posix()}")
    imageList.append("Original", originalImage)
    
    # Desaturate
    staturationScale: float = 0.5
    hsvOrignalImage: cv2.Mat = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
    hsvOrignalImage = np.float32(hsvOrignalImage)
    hueImage, satImage, valImage = cv2.split(hsvOrignalImage)
    for i in range(0, 8):
        satTempImage = np.clip(satImage * staturationScale, 0, 255)
        resultImage: cv2.Mat = np.uint8(cv2.merge([
            hueImage, 
            satTempImage,
            valImage]))
        imageList.append(f"Desat. ({staturationScale})", cv2.cvtColor(resultImage, cv2.COLOR_HSV2BGR))
        staturationScale *= 1.2

    # Show images    
    windowName = "Window"
    ShowImages(windowName, imageList)

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))