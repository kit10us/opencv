'''
    Follows documentation format from Google (https://google.github.io/styleguide/pyguide.html)
    Does not follow any specific coding standards.
'''

import sys
import os
from multiprocessing import Process
from collections.abc import Callable
from typing import override
import time

import cv2
import numpy as np


def GetFontCharacterSize(character, fontFace, fontScale, thickness):
    characterSize, _ = cv2.getTextSize(character, fontFace, fontScale, thickness)
    return characterSize

def GetFontScale(imageWidth: int, imageHeight: int, maxTextLength, fontFace:int, thickness: int):
    characterPadding: int = 2
    characterWidth: int = imageWidth / (maxTextLength + characterPadding)
    characterHeightWidthRatio: float = 1.3
    characterHeight: int = int(characterWidth * characterHeightWidthRatio)
    scale: float = cv2.getFontScaleFromHeight(fontFace, characterHeight, thickness)
    return scale

def PutOutlinedText(image: cv2.Mat, text: str, point: list[int], fontFace, fontScale, lineType):
    """ Specifying None as a point component will user image's center"""
    if len(text) == 0:
        return
    
    assert len(point)
    characterWidth: int = GetFontCharacterSize("Q", fontFace, fontScale, 1)[0]
    thickness = int(characterWidth / 12)
    outlineDeflection = int(characterWidth / 7)
    outlineCount: int = 8
    color: list = 255, 255, 255
    outlineColor: list = 0, 0, 0

    if point[0] == None:
        textWidth: int = cv2.getTextSize(text, fontFace, fontScale, thickness)[0][0]
        imageWidth: int = image.shape[1]
        point[0] = int(imageWidth / 2 - textWidth / 2)
    if point [1] == None:
        textHeight: int = cv2.getTextSize(text, fontFace, fontScale, thickness)[0][1]
        imageHeight: int = image.shape[0]
        point[1] = int(imageHeight / 2 - textHeight / 2)

    for outlineIndex in range(0, outlineCount):
        degreesRadians = np.pi * 2 / 8 * outlineIndex 
        outlinePoint = \
            point[0] + int(np.cos(degreesRadians) * outlineDeflection), \
            point[1] + int(np.sin(degreesRadians) * outlineDeflection)
        cv2.putText(image, text, outlinePoint, fontFace, fontScale, outlineColor, thickness, lineType)
    cv2.putText(image, text, point, fontFace, fontScale, color, thickness, lineType)

def DrawAlphaRectangle(image: cv2.Mat, pointUL, pointDR, alpha: int):
    clippedImage = image[pointUL[1]:pointDR[1], pointUL[0]:pointDR[0]]
    whiteRectangleImage = np.ones(clippedImage.shape, dtype=np.uint8) * alpha
    result = cv2.addWeighted(clippedImage, 0.5, whiteRectangleImage, 1.0, 1.0)
    image[pointUL[1]:pointDR[1], pointUL[0]:pointDR[0]] = result

def PasteImageRect(dest: cv2.Mat, source:cv2.Mat, ul: list[int]):
    sourceWidth = source.shape[1]
    sourceHeight = source.shape[0]
    up = ul[1]
    left = ul[0]
    down = up + sourceHeight
    right = left + sourceWidth

    finalSource: np.ndarray = source
    if len(finalSource.shape) != 3:
        finalSource = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    dest[up:down, left:right] = finalSource[0:sourceHeight, 0:sourceWidth]


def ShowCalculating(windowName: str, width: int, height: int):
    timeToTickMS: int = 250
    maxTicks: int = 4
    ticks: int = maxTicks

    calculatingImage: cv2.Mat = np.zeros([height, width], np.uint8)

    while True:
        if ticks >= maxTicks:
            ticks = 0            

        calculatingImage = cv2.rectangle(calculatingImage, (0, 0), (width, height), (0, 0, 0), cv2.FILLED)
        text: str = "Calculating, please wait"
        for t in range(0, ticks):
            text = "".join([text, "."])
        PutOutlinedText(calculatingImage, text, [None, None], cv2.FONT_HERSHEY_PLAIN, 1, cv2.LINE_AA)

        cv2.imshow(windowName, calculatingImage)
        cv2.waitKey(timeToTickMS)
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            os._exit(0)
        ticks += 1

def CreateShowCalculatingProcess(windowName: str, width: int, height: int) -> Process:
    return Process(target=ShowCalculating, args=[windowName, width, height])


class TextImage:
    _image: cv2.Mat = None
    image: cv2.Mat
    _text: str = ""
    _lastUpdated: float = 0
    _dirty: bool = True

    def __init__(self, text:str, image: cv2.Mat):
        self.setImage(image)
        self.setText(text)


    def getImage(self) -> cv2.Mat:
        return self._image
    
    def setImage(self, image: cv2.Mat):
        self._image = image.copy()
        self.__Tick()
    image = property(getImage, setImage)


    def getText(self) -> str:
        return self._text
    
    def setText(self, text:str) -> str:
        self.__Tick()
        self._text = text
    
    text = property(getText, setText)


    def getLastUpdated(self) -> float:
        return self._lastUpdated

    lastUpdated = property(getLastUpdated)


    def getIsDirty(self) -> bool:
        return self._dirty
    
    def setIsDirty(self, dirty: bool):
        self._dirty = dirty

    isDirty = property(getIsDirty, setIsDirty)


    def getSize(self) -> list[int]:
        return self.image.shape[1], self.image.shape[0]
    size = property(getSize)


    def getAspectRatio(self):
        ratio: float = 0
        imageSize: list[int] = self.size
        return imageSize[1] / imageSize[0]
    aspectRatio = property(getAspectRatio)


    def __Tick(self):
        self._lastUpdated = time.time()
        self._dirty = True

    def UpdateImage(self, input: cv2.Mat):
        self._image[:] = input[:]
        self.__Tick()
    
    def Process(self):
        '''
        Performs processing on the text image.
        '''
        pass
    
class TextMovie(TextImage):
    _video: cv2.VideoCapture = None
    _playing: bool = True
    _frameIndex: int = 0

    def __init__(self, text:str, video: cv2.VideoCapture):
        self._video = video
        success, frame = self._video.read()
        if success == False:
            raise Exception("Failed to get the frame of the video capture.")
        super().__init__(text, frame)
    
    @override
    def Process(self):
        if self.playing == False:
            return False
        
        success, frame = self._video.read()
        if success == False or frame is None:
            return False

        self.image = frame
        self._frameIndex += 1
        self.isDirty = True

    def Restart(self):
        self._playing = True
        self._video.open(0)

    def getPlaying(self):
        return self._playing
    
    def setPlaying(self, value:bool):
        self._playing = value

    def getFrameIndex(self) -> int:
        return self._frameIndex

    playing = property(getPlaying, setPlaying)
    frameIndex = property(getFrameIndex)


class ImageList:
    # Input
    __maxColumns: int = 0                     # Maximum number of columns.
    __maxWidth: int = 0                       # Maximum width for the windows.
    __maxTextLength: int = 0                  # The maximum length of image text.

    # Internal
    __imageList: list[TextImage] = None     # The list of TextImages.
    __size: list[int] = 0, 0                # Width and height of the final image list image.
    __cellSize: list[int] = 0, 0            # Width and height of the cells.
    __RCCount: list[int] = 0, 0             # Number of rows and columns.
    __scale: float = 0                      # How much the image needs to be scaled.

    def __init__(self, maxWidth: int, maxColumns: int, maxTextLength: int, imageList: list[TextImage] = None):
        """
        Args:
            maxWidth (int):         The maxuimum width we allow for the window.
            maxColumns (int):       The maximum number of columns we allow for the images.
            maxTextLength (int):    The maximum length of text we will use.
            imageList (TextImage):  A list of images to show, and their text. None skips the image cell.
        """
        self.__imageList = []

        assert maxWidth, "Max width is zero."
        assert maxColumns, "Max columns is zero."
        assert maxTextLength, "Max text length is zero."

        self.__maxWidth = maxWidth
        self.__maxColumns = maxColumns
        self.__maxTextLength = maxTextLength

        if imageList != None:
            for image in imageList:
                self.append(image)


    def append(self, first: any, second: any = None):

        textImage: TextImage = None
        if first is None:
            textImage = None
        elif type(first).__name__ == np.ndarray.__name__:
            textImage = TextImage("", first)
        elif type(first).__name__ == str.__name__ and type(second).__name__ == np.ndarray.__name__:
            textImage = TextImage(first, second)
        else:
            textImage = first

        self.__imageList.append(textImage)
        self.__CalculateInfo()


    def __CalculateInfo(self):
        """
            Calculate the information, internal variables, for this image list.
        """
        self.__size = 0, 0
        self.__cellSize = 0, 0
        self.__RCCount = 0, 0
        self.__scale = 0

        column: int = 0
        row: int = 0
        aspectRatio: float = 0.0
        aspectRatioSize: list[int] = 0, 0
        largestWidth: int = 0
        index: int = 0
        for image in self.__imageList:
            if image is None:
                continue

            largestWidth = np.max([largestWidth, image.size[0]])

            if image.aspectRatio > aspectRatio:
                aspectRatio = image.aspectRatio
                aspectRatioSize = image.size

        # Deterine number of rows
        columns: int = np.min([len(self.__imageList), self.__maxColumns])
        self.__RCCount = int(np.ceil(len(self.__imageList) / columns)), columns

        # Determine scale
        self.__scale = np.round(self.__maxWidth / (largestWidth * columns), 5)

        # Determine real cell size
        cellWidth: float =  self.__maxWidth / self.__RCCount[1]
        cellHeight: float = (cellWidth / aspectRatioSize[0]) * aspectRatioSize[1]

        self.__cellSize = int(cellWidth), int(cellHeight)
        self.__size = self.__cellSize[0] * self.__RCCount[1], self.__cellSize[1] * self.__RCCount[0]


    def getRCCount(self) -> list[int]:
        return self.__RCCount
    RCCount = property(getRCCount)


    def getSize(self) -> list[int]:
        return self.__size
    size = property(getSize)


    def RC(self, imageIndex: int) -> list[int]:
        row: int = int(np.floor(imageIndex / self.RCCount[1]))
        column:int = imageIndex - row * self.RCCount[1]
        return row, column

    def getCellSize(self) -> list[int]:
        return self.__cellSize
    cellSize = property(getCellSize)


    def UL(self, imageIndex: int) -> list[int]:
        return int(self.RC(imageIndex)[1] * self.cellSize[0]), int(self.RC(imageIndex)[0] * self.cellSize[1])
    
    
    def DR(self, imageIndex: int):
        return self.UL(imageIndex) + self.getCellSize() - (1, 1)
    

    def getScale(self) -> float:
        return self.__scale
    scale = property(getScale)

    
    def imageScaledSize(self, index: int) -> list[int]:
        assert index < len(self.__imageList), "Image index out of range."
        if self.__imageList[index] == None:
            return self.__cellSize
        
        scale: float = 0
        imageWidth, imageHeight = self.__imageList[index].size
        imageScale: float = imageHeight / imageWidth
        if imageWidth > imageHeight:
            scale = self.__cellSize[0] / imageWidth
        else:
            scale = self.__cellSize[1] / imageHeight
        return int(imageWidth * scale), int(imageHeight * scale)
    

    def getScaledImage(self, index: int) -> cv2.Mat:
        assert index < len(self.__imageList), "Image index out of range."
        if self.__imageList[index] == None:
            return None 

        image: cv2.Mat = self.__imageList[index].image
        scaledImage = image.copy()
        scaledImage = cv2.resize(scaledImage, self.imageScaledSize(index))
        tempImage = scaledImage.copy()
        scaledImage = np.zeros([self.__cellSize[1], self.__cellSize[0], 3], dtype=np.uint8)
        PasteImageRect(scaledImage, tempImage, (0, 0))

        # Add text to full image
        lineType = cv2.LINE_AA        
        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontThickness = 2
        text: str = self.__imageList[index].text
        fontScale = GetFontScale(self.__cellSize[0], self.__cellSize[1], self.__maxTextLength, fontFace, fontThickness)
        textWidth, textHeight = cv2.getTextSize(text, fontFace, fontScale, fontThickness)[0]

        point = int(self.__cellSize[0] / 2 - textWidth / 2), int(self.__cellSize[1] - textHeight * 1)
        PutOutlinedText(scaledImage, text, point, fontFace, fontScale, lineType)
        return scaledImage


    def BuildDisplayImage(self) -> cv2.Mat:    #maxWidth: int, maxColumns: int, maxTextLength: int, imageList: list[TextImage]) -> cv2.Mat:
        # Initialize values
        imageOutput: cv2.Mat = np.zeros((self.size[1], self.size[0], 3), dtype = np.uint8)

        imageIndex: int = 0
        for textImage in self.__imageList:
            if textImage != None:
                scaledImage: np.mat = self.getScaledImage(imageIndex)
                assert scaledImage is not None

                ul: list[int] = self.UL(imageIndex)

                # Attempt to paste the image as if it were a BGR image, else attempt a gray image.
                try:
                    PasteImageRect(imageOutput, scaledImage, ul)
                except:
                    tempColorImage: cv2.Mat = cv2.cvtColor(scaledImage, cv2.COLOR_GRAY2BGR)
                    PasteImageRect(imageOutput, tempColorImage, ul)

            imageIndex += 1
        return imageOutput
    
    def __len__(self) -> int:
        return len(self.__imageList)
    
    def __getitem__(self, index) -> TextImage:
        if index > len(self.__imageList):
            return None
        self.__ran
        return self.__imageList[index]
    
    def __iter__(self) -> TextImage:
        for textImage in self.__imageList:
            yield textImage


def ShowImages(windowName: str, imageList: ImageList, process: callable = None):
    finalImage: cv2.Mat
    rows: int
    columns: int
    fontScale: float

    finalImage = imageList.BuildDisplayImage()

    imageHeight, imageWidth = finalImage.shape[0:2]
    imageCount: int = len(imageList)

    def MouseCallback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            imageColumnClicked = int(x / imageWidth * columns)
            imageRowClicked = int(y / imageHeight * rows)
            imageIndex = imageColumnClicked + imageRowClicked * columns
            if imageIndex < imageCount:
                singleImageWindow = "Single"
                image = imageList[imageIndex]
                cv2.imshow(singleImageWindow, image)
    
    instructionTexts = ["Close window or press any key to exit", "Click on an image to see it fully"]
    showInstructionsDuration = 8000
    currentDuration = 0
    showInstructions = False
    hasShownFinalWindow = False
    instructionImage = None
    
    if showInstructions and showInstructionsDuration > 0 and instructionTexts != None:
        instructionFontThickness = 1
        charactersOnHorizontal = 70
        charactersInBox = 30
        fontFace = cv2.FONT_HERSHEY_PLAIN
        instructionFontScale = GetFontScale(imageWidth, imageHeight, charactersOnHorizontal, fontFace, instructionFontThickness)
        characterWidth, characterHeight = GetFontCharacterSize("Q", fontFace, instructionFontScale, instructionFontThickness)
        point = characterHeight * 2, characterHeight * 3
        textSpacing = int(np.ceil(0.4 * characterHeight))
        newlineHeight = textSpacing + characterHeight
        totalTextHeight = int(newlineHeight * len(instructionTexts))
        pointUL = point[0] - int(characterHeight * 1.5), int(point[1] - characterHeight * 1.5)
        pointDR = point[0] + int(charactersInBox * characterWidth), point[1] + totalTextHeight

    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, MouseCallback)

    runLoop = True
    while runLoop:
        cv2.imshow(windowName, finalImage)
        cv2.waitKey(1)

        if showInstructions == True:
            if currentDuration < showInstructionsDuration:
                #showInstructions = False
                instructionImage = np.copy(finalImage)        
                DrawAlphaRectangle(instructionImage, pointUL, pointDR, 155)
                for i in range(0, len(instructionTexts)):
                    text = instructionTexts[i]
                    PutOutlinedText(instructionImage, text, point, fontFace, fontScale,cv2.LINE_AA)
                    point = point[0], point[1] + newlineHeight
                    cv2.imshow(windowName, instructionImage)
                    cv2.waitKey(1)

        elif hasShownFinalWindow == False:
            hasShownFinalWindow = True
            cv2.imshow(windowName, finalImage)
            cv2.waitKey(1)

        if process != None:
            result: bool = process(imageList)
            if result == False:
                return

        textImage:TextImage
        changed: int = 0
        for textImage in imageList:
            if textImage != None:
                textImage.Process()
                if textImage.isDirty:
                    changed += 1
                    textImage.isDirty = False

        if changed > 0:
            finalImage = imageList.BuildDisplayImage()
        
        cv2.waitKey(100)
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            runLoop = False
    cv2.destroyAllWindows()


def UnitTest(argc: int, argv: list[str]) -> int:
    def RunTestSuite(test: Callable):
        print(f"Starting test suite {test.__name__}")
        test()
        print(f"End test suite {test.__name__} passed")
        print()
    
    def Expect(name:str, got: any, expected: any):
        output: str = ""
        if got == expected:
            print(f"Test {name} passed.")
        else:
            output: str = f"Test {name} failed. {got = }, {expected = }"
            print(output)
            assert got == expected, output

    def timeout(imageList: ImageList):
        current: float = time.time()
        if timeout.first == 0:
            timeout.first = current
        duration = current - timeout.first
        if duration >= timeout.timeout:
            return False
        return True
    timeout.timeout = 3
    timeout.first = 0


    def TestOneImage():
        maxWith: int = 400
        maxColumns: int = 2
        maxTextLength: int = 32
        imageList: ImageList = ImageList(maxWith, maxColumns, maxTextLength)

        component: int = 255

        image: np.ndarray = np.zeros([200, 600, 3], dtype = np.uint8)
        image[:][:] = [component, 0, 0]
        imageList.append("Test", image)

        expectedRCCount: list[int] = 1, 1
        Expect("Row and column count", imageList.RCCount, expectedRCCount)

        expectedRC0: list[int] = 0, 0
        Expect("Image 0 RC", imageList.RC(0), expectedRC0)

        expectedCellSize: list[int] = 400, 133
        Expect("Cell size", imageList.cellSize, expectedCellSize)

        expectedSize: list[int] = 400, 133
        Expect("Size", imageList.size, expectedSize)

        expectedScale: float = 0.66666666
        Expect("Scale", np.round(imageList.scale, 5), np.round(expectedScale, 5))

        expectedUL0: list[int] = 0, 0
        Expect("Image 0 UL", imageList.UL(0), expectedUL0)

        expectedScaledSize0: list[int] = 400, 133
        Expect("Image 0 scaled size", imageList.imageScaledSize(0), expectedScaledSize0)

        timeout.timeout = 3
        timeout.first = 0
        ShowImages("Test", imageList, timeout)        


    def TestTwoImagesFirstMaxWidthSecondMaxHeight():
        maxWith: int = 400
        maxColumns: int = 2
        maxTextLength: int = 32
        imageList: ImageList = ImageList(maxWith, maxColumns, maxTextLength)

        component: int = 255

        image1: np.ndarray = np.zeros([200, 600, 3], dtype = np.uint8)
        image1[:][:] = [component, 0, 0]
        imageList.append("Test", image1)

        image2: np.ndarray = np.zeros([600, 200, 3], dtype = np.uint8)
        image2[:][:] = [0, component,0]
        imageList.append("Test", image2)

        expectedRCCount: list[int] = 1, 2
        Expect("Row and column count", imageList.RCCount, expectedRCCount)

        expectedRC0: list[int] = 0, 0
        Expect("Image 0 RC", imageList.RC(0), expectedRC0)

        expectedRC1: list[int] = 0, 1
        Expect("Image 1 RC", imageList.RC(1), expectedRC1)

        expectedCellSize: list[int] = 200, 600
        Expect("Cell size", imageList.cellSize, expectedCellSize)

        expectedSize: list[int] = 400, 600
        Expect("Size", imageList.size, expectedSize)

        expectedUL0: list[int] = 0, 0
        Expect("Image 0 UL", imageList.UL(0), expectedUL0)

        expectedUL1: list[int] = expectedCellSize[0] * 1, 0
        Expect("Image 1 UL", imageList.UL(1), expectedUL1)

        expectedScaledSize0: list[int] = 200, 66
        Expect("Image 0 scaled size", imageList.imageScaledSize(0), expectedScaledSize0)

        expectedScaledSize1: list[int] = 200, 600
        Expect("Image 1 scaled size", imageList.imageScaledSize(1), expectedScaledSize1)

        timeout.timeout = 3
        timeout.first = 0
        ShowImages("Test", imageList, timeout)


    def TestFourImages():
        maxWith: int = 500
        maxColumns: int = 2
        maxTextLength: int = 32
        imageList: ImageList = ImageList(maxWith, maxColumns, maxTextLength)

        component: int = 255

        image1: np.ndarray = np.zeros([200, 300, 3], dtype = np.uint8)
        image1[:][:] = [component, 0, 0]
        imageList.append("Test", image1)

        image2: np.ndarray = np.zeros([300, 200, 3], dtype = np.uint8)
        image2[:][:] = [0, component,0]
        imageList.append("Test", image2)

        image3: np.ndarray = np.zeros([300, 200, 3], dtype = np.uint8)
        image3[:][:] = [0, 0, component]
        imageList.append("Test", image3)

        image4: np.ndarray = np.zeros([200, 300, 3], dtype = np.uint8)
        image4[:][:] = [component, component,0]
        imageList.append("Test", image4)

        expectedRCCount: list[int] = 2, 2
        Expect("Row and column count", imageList.RCCount, expectedRCCount)

        expectedRC0: list[int] = 0, 0
        Expect("Image 0 RC", imageList.RC(0), expectedRC0)

        expectedRC1: list[int] = 0, 1
        Expect("Image 1 RC", imageList.RC(1), expectedRC1)

        expectedCellSize: list[int] = 250, 375
        Expect("Cell size", imageList.cellSize, expectedCellSize)

        expectedSize: list[int] = 500, 750
        Expect("Size", imageList.size, expectedSize)

        expectedUL0: list[int] = 0, 0
        Expect("Image 0 UL", imageList.UL(0), expectedUL0)

        expectedUL1: list[int] = expectedCellSize[0] * 1, 0
        Expect("Image 1 UL", imageList.UL(1), expectedUL1)

        expectedScaledSize0: list[int] = 250, 166
        Expect("Image 0 scaled size", imageList.imageScaledSize(0), expectedScaledSize0)

        expectedScaledSize1: list[int] = 250, 375
        Expect("Image 1 scaled size", imageList.imageScaledSize(1), expectedScaledSize1)

        expectedScaledSize2: list[int] = 250, 375
        Expect("Image 2 scaled size", imageList.imageScaledSize(2), expectedScaledSize2)

        expectedScaledSize3: list[int] = 250, 166
        Expect("Image 3 scaled size", imageList.imageScaledSize(3), expectedScaledSize3)

        timeout.timeout = 3
        timeout.first = 0
        ShowImages("Test", imageList, timeout)

    def TestTwoImagesFailedCaseSkyPlane():
        maxWith: int = 1300
        maxColumns: int = 2
        maxTextLength: int = 20
        imageList: ImageList = ImageList(maxWith, maxColumns, maxTextLength)

        component: int = 255

        image1: np.ndarray = np.zeros([561, 1000, 3], dtype = np.uint8)
        image1[:][:] = [component, 0, 0]
        imageList.append("Test", image1)

        image2: np.ndarray = np.zeros([194, 300, 3], dtype = np.uint8)
        image2[:][:] = [0, component,0]
        imageList.append("Test", image2)

        expectedRCCount: list[int] = 1, 2
        Expect("Row and column count", imageList.RCCount, expectedRCCount)

        expectedRC0: list[int] = 0, 0
        Expect("Image 0 RC", imageList.RC(0), expectedRC0)

        expectedRC1: list[int] = 0, 1
        Expect("Image 1 RC", imageList.RC(1), expectedRC1)

        expectedCellSize: list[int] = 650, 420
        Expect("Cell size", imageList.cellSize, expectedCellSize)

        expectedSize: list[int] = 1300, 420
        Expect("Size", imageList.size, expectedSize)

        expectedUL0: list[int] = 0, 0
        Expect("Image 0 UL", imageList.UL(0), expectedUL0)

        expectedUL1: list[int] = expectedCellSize[0] * 1, 0
        Expect("Image 1 UL", imageList.UL(1), expectedUL1)

        expectedScaledSize0: list[int] = 650, 364
        Expect("Image 0 scaled size", imageList.imageScaledSize(0), expectedScaledSize0)

        expectedScaledSize1: list[int] = 650, 420
        Expect("Image 1 scaled size", imageList.imageScaledSize(1), expectedScaledSize1)

        timeout.timeout = 3
        timeout.first = 0
        ShowImages("Test", imageList, timeout)

    def TestGreyImage():
        maxWith: int = 400
        maxColumns: int = 2
        maxTextLength: int = 32
        imageList: ImageList = ImageList(maxWith, maxColumns, maxTextLength)

        component: int = 155

        image: np.ndarray = np.zeros([200, 600, 1], dtype = np.uint8)
        image[:][:] = [component]
        imageList.append("Test", image)

        expectedRCCount: list[int] = 1, 1
        Expect("Row and column count", imageList.RCCount, expectedRCCount)

        expectedRC0: list[int] = 0, 0
        Expect("Image 0 RC", imageList.RC(0), expectedRC0)

        expectedCellSize: list[int] = 400, 133
        Expect("Cell size", imageList.cellSize, expectedCellSize)

        expectedSize: list[int] = 400, 133
        Expect("Size", imageList.size, expectedSize)

        expectedScale: float = 0.66666666
        Expect("Scale", np.round(imageList.scale, 5), np.round(expectedScale, 5))

        expectedUL0: list[int] = 0, 0
        Expect("Image 0 UL", imageList.UL(0), expectedUL0)

        expectedScaledSize0: list[int] = 400, 133
        Expect("Image 0 scaled size", imageList.imageScaledSize(0), expectedScaledSize0)

        timeout.timeout = 3
        timeout.first = 0
        ShowImages("Test", imageList, timeout)        

    RunTestSuite(TestOneImage)
    RunTestSuite(TestTwoImagesFirstMaxWidthSecondMaxHeight)
    RunTestSuite(TestFourImages)
    RunTestSuite(TestTwoImagesFailedCaseSkyPlane)
    RunTestSuite(TestGreyImage)

    return 0

if __name__ == "__main__":
    exit(UnitTest(len(sys.argv), sys.argv))