import sys
import cv2
import numpy as np
import typing
from pathlib import Path

import ss_showimages as ss

from ss_util import GetAssetPath
from ss_util import AssetType
from ss_showimages import TextImage
from ss_showimages import TextMovie
from ss_showimages import ShowImages, ImageList
from ss_rect import Rect

def var_abs_laplacian_var(image: cv2.Mat, intermediate: cv2.Mat, rect: Rect) -> float:
    imageGrayscale: cv2.Mat
    laplacianImage: cv2.Mat

    if rect == None:
        rect = Rect([0, 0], [image.shape[1] - 1, image.shape[0] - 1])
    imageGrayscale = cv2.cvtColor(rect.crop(image), cv2.COLOR_BGR2GRAY)

    laplacianImage = cv2.Laplacian(imageGrayscale, cv2.CV_64F)
    focusedValue: float = laplacianImage.var()
    
    imageIn256 = np.zeros_like(laplacianImage, np.uint8)
    imageIn256.flat[:] = laplacianImage.flat[:] * 255

    out: cv2.Mat
    out = cv2.cvtColor(imageIn256, cv2.COLOR_GRAY2BGR)
    intermediate[:] = out[:]
    cv2.rectangle(intermediate, rect.ul, rect.dr, (0, 0, 255), 3)

    return focusedValue


def var_abs_laplacian_summod(image: cv2.Mat, intermediate: cv2.Mat, rect: Rect) -> float:
    if rect == None:
        rect = Rect([0, 0], [image.shape[1] - 1, image.shape[0] - 1])
    imageGrayscale = cv2.cvtColor(rect.crop(image), cv2.COLOR_BGR2GRAY)

    kernelA: np.mat = np.array([[-1, 2, -1]])
    filtered: cv2.Mat = cv2.filter2D(imageGrayscale, -1, kernelA)

    kernelB: np.mat = np.array([[-1], [2], [-1]])
    filtered: cv2.Mat = cv2.filter2D(filtered, -1, kernelB)

    focusedValue: any = np.sum(filtered[:, :])

    out: cv2.Mat
    out = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    rect.paste(out, intermediate)
    cv2.rectangle(intermediate, rect.ul, rect.dr, (0, 0, 255), 3)
    return focusedValue

class ABSLaplacian(TextImage):
    _source: TextMovie
    _sourceUpdated: float
    _bestFrame: TextImage = None
    _worstFrame: TextImage = None

    _rect: Rect

    bestSML: float = -10000
    bestFrameIndex: int = 0
    worstSML: float = 10000
    worstFrameIndex: int = 0

    lapFunc = None 

    def __init__(self, lapFun, source: TextImage, bestFrame: TextImage, worstFrame: TextImage, rect: Rect = None):
        self.lapFunc = lapFun
        self.__rect = rect
        self._source = source
        self._sourceUpdated = 0
        super().__init__("ABS Laplacian", source.image.copy())
        self._bestFrame = bestFrame
        self._worstFrame = worstFrame
        self._bestFrame.setImage(source.getImage().copy())
        self._worstFrame.setImage(source.getImage().copy())
        self.Update()

    def Update(self):
        newTime: float = self._source.lastUpdated
        
        if newTime == self._sourceUpdated:
            return False
        
        self._sourceUpdated = newTime

        if self.lapFunc == None:
            return
        
        image: cv2.Mat = self._source.getImage()
        intermediate: cv2.Mat = image.copy()
        SML:float = self.lapFunc(image, intermediate, self.__rect)
        self.UpdateImage(intermediate)

        frameIndex:int = self._source.frameIndex
        if SML > self.bestSML:
            self._bestFrame.UpdateImage(image)
            self.bestFrameIndex = frameIndex
            self.bestSML = SML
            print(f"{self.lapFunc.__name__ = } = {self.bestFrameIndex = }, {self.bestSML = } ... {self.worstFrameIndex = }, {self.worstSML = }")

        if SML < self.worstSML:
            self._worstFrame.UpdateImage(image)
            self.worstFrameIndex = frameIndex
            self.worstSML = SML
            print(f"{self.lapFunc.__name__ = } = {self.bestFrameIndex = }, {self.bestSML = } ... {self.worstFrameIndex = }, {self.worstSML = }")
        self.isDirty = True

    @typing.override
    def Process(self):
        self.Update()


def main(argc: int, argv: list[str]) -> int:
    videoPath = GetAssetPath("focus-test.mp4", AssetType.Video)
    video = cv2.VideoCapture(videoPath.as_posix())
    if video.isOpened() == False:
        raise Exception(f"Failed to open image {videoPath}")
    
    imageList: ImageList = ImageList(1500, 3, 30)
    movie = TextMovie("Movie", video)
    imageList.append(movie)
    imageList.append(None)
    imageList.append(None)

    bestFrame1: TextImage = TextImage("Best Frame 1", movie.getImage().copy())
    worstFrame1: TextImage = TextImage("Worst Frame 1", movie.getImage().copy())
    imageList.append(ABSLaplacian(var_abs_laplacian_var, movie, bestFrame1, worstFrame1))
    imageList.append(bestFrame1);
    imageList.append(worstFrame1)

    bestFrame2: TextImage = TextImage("Best Frame 2", movie.getImage().copy())
    worstFrame2: TextImage = TextImage("Worst Frame 2", movie.getImage().copy())
    imageList.append(ABSLaplacian(var_abs_laplacian_summod, movie, bestFrame2, worstFrame2, Rect([550, 150], [950, 550])))
    imageList.append(bestFrame2);
    imageList.append(worstFrame2)

    ShowImages("Autofocus", imageList)

    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))
