import sys
from pathlib import Path

import cv2
import numpy as np

from ss_util import GetAssetPath
from ss_util import AssetType
from ss_showimages import ImageList
from ss_showimages import ShowImages

def main(argc: int, argv: list[str]) -> int:
    def FloatImageToUint8(floatImage: cv2.Mat) -> cv2.Mat:
        uint8Image = np.zeros_like(floatImage)#, np.uint8)
        uint8Image[:,:] = floatImage[:,:] * 256
        uint8Image = uint8Image.astype(np.uint8)
        return uint8Image
    
    windowName = "Filtering"
    imageList: ImageList = ImageList(1024, 4, 20)    

    # Noise bluring
    noisyImagePath = GetAssetPath("boy_noisy.jpg", AssetType.Image)
    noisyImage: cv2.Mat = cv2.imread(noisyImagePath.as_posix())
    if noisyImage is None:
        raise Exception(f"Unable to read image {noisyImagePath}")
    imageList.append("Noisy", noisyImage)

    medianImage: cv2.Mat = cv2.medianBlur(noisyImage, 3,)
    imageList.append("Median", medianImage)

    bilateralImage: cv2.Mat = cv2.bilateralFilter(noisyImage, 15, 80, 80) 
    imageList.append("Bilateral", bilateralImage)

    convolveKernel = np.zeros([3, 3], np.float32)
    convolveKernel[0,:] = [ 0, -1,  0]
    convolveKernel[1,:] = [-1,  4, -1]
    convolveKernel[2,:] = [ 0, -1,  0]
    convolvedImage: cv2.Mat = cv2.filter2D(noisyImage, -1, convolveKernel) 
    imageList.append("Convolved", convolvedImage)

    # First order derivative filters
    grayFOImagePath: Path = GetAssetPath("truth.png", AssetType.Image)
    grayFOImage: cv2.Mat = cv2.imread(grayFOImagePath.as_posix(), cv2.IMREAD_GRAYSCALE)
    if grayFOImage is None:
        raise Exception(f"Unable to read image {grayFOImagePath}")
    imageList.append("Gray", grayFOImage)
    
    sobelDepth = cv2.CV_32F

    sobelXImage: cv2.Mat = cv2.Sobel(grayFOImage, sobelDepth, 1, 0)
    cv2.normalize(sobelXImage, sobelXImage, 0, 1, cv2.NORM_MINMAX, sobelDepth)
    sobelXFinalImage = FloatImageToUint8(sobelXImage)
    imageList.append("Sobel X", sobelXFinalImage)

    sobelYImage: cv2.Mat = cv2.Sobel(grayFOImage, sobelDepth, 0, 1)
    cv2.normalize(sobelYImage, sobelYImage, 0, 1, cv2.NORM_MINMAX, sobelDepth)
    sobelYFinalImage = FloatImageToUint8(sobelYImage)
    imageList.append("Sobel Y", sobelYFinalImage)

    sobelXYImage: cv2.Mat = cv2.Sobel(grayFOImage, sobelDepth, 1, 1)
    cv2.normalize(sobelXYImage, sobelXYImage, 0, 1, cv2.NORM_MINMAX, sobelDepth)
    sobelXYFinalImage = FloatImageToUint8(sobelXYImage)
    imageList.append("Sobel X & Y", sobelXYFinalImage)


    # Second order derivaitves filters
    graySOImagePath = GetAssetPath("sample.jpg", AssetType.Image)
    graySOImage: cv2.Mat = cv2.imread(graySOImagePath.as_posix(), cv2.IMREAD_GRAYSCALE)
    if graySOImage is None:
        raise Exception(f"Unable to read image {graySOImagePath}")
    imageList.append("Original", graySOImage)
    
    laplacianDepth: int = cv2.CV_32F
    laplacianKernelSize: int = 3

    laplacianImage = cv2.Laplacian(graySOImage, laplacianDepth, ksize=laplacianKernelSize, scale=1, delta=0)
    cv2.normalize(laplacianImage, laplacianImage, 0, 1, cv2.NORM_MINMAX, laplacianDepth)
    laplacianFinalImage = FloatImageToUint8(laplacianImage)
    imageList.append("Laplacian", laplacianFinalImage)

    laplacianBlurredImage = cv2.GaussianBlur(graySOImage, (laplacianKernelSize, laplacianKernelSize), 0, 0)
    laplacianGaussedBlurredImage = cv2.Laplacian(laplacianBlurredImage, laplacianDepth, ksize=laplacianKernelSize, scale=1, delta=0)
    cv2.normalize(laplacianGaussedBlurredImage, laplacianGaussedBlurredImage, 0, 1, cv2.NORM_MINMAX, laplacianDepth)
    laplacianBlurredFinaleImage = FloatImageToUint8(laplacianGaussedBlurredImage)
    imageList.append("Laplacian Blurred", laplacianBlurredFinaleImage)

    ShowImages(windowName, imageList)
    

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))