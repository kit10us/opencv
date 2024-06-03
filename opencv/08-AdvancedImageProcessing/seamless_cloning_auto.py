import sys
from typing import Sequence

import cv2
import numpy as np

from ss_util import GetAssetPath
from ss_util import AssetType
from ss_showimages import TextImage
from ss_showimages import ImageList
from ss_showimages import ShowImages


class AutoContour(TextImage):
    __imageSource: np.ndarray = None
    maxMorphCount: int = None
    targetCount: int = None
    targetContourLength: int = None
    minThreshold: int = None
    maxThreshold: int = None
    originalText: str = ""

    morphStrength: int = 3
    maxMorphStrength: int = 7
    morphCount: int = 0
    threshold: int = 10
    maxVal: int = 255
    _contours: Sequence[np.ndarray] = []
    __imageGray: np.ndarray = None
    __imageMorphed: np.ndarray = None
    __imageBinary: np.ndarray = None
    color: list[int] = (255, 0, 0)
    thickness: int = 3

    MAX: int = 999
    bestContourCount: int = MAX
    bestContourLen: int = MAX

    ''' Generates a contour based on an expected contour count. '''
    def __init__(self, text: str, imageSource: np.ndarray, targetCount: int, targetContourLength: int, minThreshold: int = 10, maxThreshold: int = 255, maxMorphCount: int = 3, maxMorphStrength: int = 3):
        super().__init__(text, imageSource)
        self.originalText = text
        self.__imageSource = imageSource.copy()
        self.Restart(targetCount, targetContourLength, minThreshold, maxThreshold, maxMorphCount, maxMorphStrength)

    def Restart(self, targetCount: int = None, targetContourLength: int = None, minThreshold: int = None, maxThreshold: int = None, maxMorphCount: int = None, maxMorphStrength: int = None):
        assert self.__imageSource is not None

        self.status: str = "Not Started"
        if targetCount != None:
            self.targetCount = targetCount

        if targetContourLength != None:
            self.targetContourLength = targetContourLength

        if minThreshold != None:
            self.minThreshold = minThreshold

        if maxThreshold != None:
            self.maxThreshold = maxThreshold

        if maxMorphCount != None:
            self.maxMorphCount = maxMorphCount
        
        if maxMorphStrength != None:
            self.maxMorpthStrength = maxMorphStrength

        assert (self.targetCount != None and self.targetContourLength != None and self.maxThreshold != None and self.maxMorphCount != None)

        self.threshold = self.minThreshold

        self._contours = []
        self.morphCount = 0
        #self.__imageGray = cv2.cvtColor(self.__imageSource, cv2.COLOR_BGR2GRAY)
        self.__generateGrayImage()
        self.__imageMorphed = self.imageGray.copy()
        self.__imageBinary = np.zeros_like(self.imageGray)
        self.__imageContoured = self.image
        self.bestContourCount = self.MAX
        self.bestContourLen = self.MAX


    def getContours(self) -> Sequence[np.ndarray]:
        return self._contours
    contours = property(getContours)


    def getImageSource(self) -> np.ndarray:
        return self.__imageSource     
    imageSource = property(getImageSource)


    def getImageGray(self) -> np.ndarray:
        return self.__imageGray     
    imageGray = property(getImageGray)


    def getImageBinary(self) -> np.ndarray:
        return self.__imageBinary     
    imageBinary = property(getImageBinary)

    
    def getImageMorphed(self) -> np.ndarray:
        return self.__imageMorphed     
    imageMorphed = property(getImageMorphed)
    

    def getImageContoured(self) -> np.ndarray:
        return self.__imageContoured     
    imageContoured = property(getImageContoured)


    def __generateGrayImage(self):
        self.__imageGray = cv2.cvtColor(self.__imageSource, cv2.COLOR_BGR2GRAY)
        for y in range(self.__imageSource.shape[0]):
            for x in range(self.__imageSource.shape[1]):
                red: int = self.__imageSource[y, x, 2]
                green: int = self.__imageSource[y, x, 1]
                blue: int = self.__imageSource[y, x, 0]
                gray: int = np.round(red * 0.299 + green * 0.587 + blue * 0.114)
                self.__imageGray[y, x] = gray
    

    # class TextImage
    
    def Process(self):
        def SetStatus(status: str):
            self.text = f"{status} ms: {self.morphStrength} (c {len(self.contours)}, bc {self.bestContourCount}, bcl {self.bestContourLen})"

        # Check if we need to give up.
        if self.threshold >= self.maxThreshold:
            if self.morphCount >= self.maxMorphCount:
                if self.morphStrength >= self.maxMorphStrength:
                    SetStatus("Failed")
                    return
                self.morphStrength += 2
            self.morphCount += 1
            self.threshold = self.minThreshold
            kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [self.morphStrength, self.morphStrength])
            self.__imageMorphed = cv2.morphologyEx(self.__imageMorphed, cv2.MORPH_DILATE, kernel)

        _, self.__imageBinary = cv2.threshold(self.__imageMorphed, thresh=self.threshold, maxval=self.maxVal, type=cv2.THRESH_BINARY)

        self._contours, _ = cv2.findContours(self.__imageBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contourCount: int = len(self.contours)
        contourCountLen: int = self.MAX
        for contour in self.contours:
            contourCountLen = np.min([contourCountLen, len(contour)])

        if contourCount >= self.targetCount:
            if contourCount < self.bestContourCount:
                self.bestContourCount = contourCount
                self.bestContourLen = contourCountLen
            elif contourCount == self.bestContourCount:
                if contourCountLen < self.bestContourLen:
                    self.bestContourCount = contourCountLen
            if contourCount == self.targetCount and self.bestContourLen <= self.targetContourLength:
                SetStatus("Completed")
                return
        SetStatus("working")

        self.__imageContoured = self.imageSource.copy()
        cv2.drawContours(self.__imageContoured, self.contours, contourIdx=-1, color=self.color, thickness=self.thickness)

        imageOutput: np.ndarray = np.zeros_like(self.imageSource)
        def UpdateQuarter(dest: np.ndarray, source: np.ndarray, quarter: list[int]):
            quarterSize: list[int] = int(np.floor(dest.shape[1] * .5)), int(np.floor(dest.shape[0] * .5))
            ul: list[int] = np.multiply(quarterSize, quarter)
            dr: list[int] = np.add(ul, quarterSize)
            if len(source.shape) == 2:
                dest[ul[1]:dr[1], ul[0]:dr[0], 0] = cv2.resize(source, quarterSize)[:,:]
                dest[ul[1]:dr[1], ul[0]:dr[0], 1] = cv2.resize(source, quarterSize)[:,:]
                dest[ul[1]:dr[1], ul[0]:dr[0], 2] = cv2.resize(source, quarterSize)[:,:]
            else:
                dest[ul[1]:dr[1], ul[0]:dr[0]] = cv2.resize(source, quarterSize)[:,:,:]

        UpdateQuarter(imageOutput, self.imageGray, [0, 0])
        UpdateQuarter(imageOutput, self.imageMorphed, [1, 0])
        UpdateQuarter(imageOutput, self.imageBinary, [0, 1])
        UpdateQuarter(imageOutput, self.imageContoured, [1, 1])


        self.UpdateImage(imageOutput)
       
        self.threshold += 1
        
        self.isDirty = True
        return super().Process()
    

def main(argc: int, argv: list[str]) -> int:
    imageList: ImageList = ImageList(800, 2, 40)

    imageBackground: np.ndarray = cv2.imread(GetAssetPath("sky.jpg", AssetType.Image).as_posix())
    assert imageBackground is not None, "Unable to load image."
    imageList.append("Sky", imageBackground)

    imagePlane: np.ndarray = cv2.imread(GetAssetPath("airplane.jpg", AssetType.Image).as_posix())
    assert imagePlane is not None, "Unable to load image."
    imageList.append("Plane", imagePlane)

    contourObject: AutoContour = AutoContour("Contoured", imagePlane, 1, 10, 140, 200, 10, 3)
    imageList.append(contourObject)

    ShowImages("Cloning", imageList)

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))

