import sys
from pathlib import Path
import cv2
import numpy as np
import time
from typing import List
from threading import Thread
from threading import Lock

def TimeNSToMicroS(timeNS) -> float:
    return timeNS * 0.001

def TimeNSToMS(timeNS) -> float:
    return TimeNSToMicroS(timeNS) * 0.001

def TimeNSToS(timeNS) -> float:
    return TimeNSToMS(timeNS) * 0.001

def GetImagePath(filename) -> Path:
    return Path(sys.argv[0]).parent.parent / "images" / filename

def LoadImage(path: Path) -> cv2.Mat:
    if path.exists() == False:
        raise FileNotFoundError("Unable to find file " + path.as_posix())
        
    image = cv2.imread(path.as_posix())
    if image is None:
        raise Exception("File found, " + path.as_posix() + ", however, unable to read")
    return image

def GetFontCharacterSize(character, fontFace, fontScale, thickness):
    characterSize, _ = cv2.getTextSize(character, fontFace, fontScale, thickness)
    return characterSize

def GetFontScale_old(maxTextCharacters: int, areaWidth: int, fontFace: int, thickness: int):
    maxTextExample: str = "Q" * maxTextCharacters
    exampleTextSize, _ = cv2.getTextSize(maxTextExample, fontFace, 1, thickness)
    return (areaWidth * 0.95) / exampleTextSize[0]

def GetFontScale(imageWidth, maxTextLength, fontFace:int, thickness: int):
    character: str = "Q"
    wantedCharacterWidth = imageWidth / maxTextLength
    f: float = 1
    characterWidth = 0
    errorValue = 1.0
    while True:
        characterSize, _ = cv2.getTextSize(character, fontFace, f, thickness)
        characterWidth = characterSize[0]
        if characterWidth < wantedCharacterWidth - errorValue:
            f *= wantedCharacterWidth / characterWidth
        elif characterWidth > wantedCharacterWidth + errorValue:
            f *= characterWidth / wantedCharacterWidth
        else:
            break
    return f

def PutOutlinedText(image, text, point, fontFace, fontScale, outlineDeflection, color, outlineColor, thickness, lineType):
    outlineCount: int = 8
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

def FillImageRect(dest: cv2.Mat, source, ulx:int, uly:int, drx: int, dry: int):
    dest[uly:dry, ulx:drx] = source[:, :]

class ImageProcess:
    lastProcessTime: int = 0
    lastUpdateTime: int = 0

    def __init__(self):
        self.UpdateProcessTime()
        self.SetLastUpdateTime()

    def Process(self):
        self.UpdateProcessTime()

    def UpdateProcessTime(self):
        self.lastProcessTime = time.time_ns()

    def SetLastUpdateTime(self):
        self.lastUpdateTime = time.time_ns()

    def GetProcessTime(self) -> int:
        return self.lastProcessTime
    
    def GetLastUpdateTime(self):
        return self.lastUpdateTime

    def GetRawResults(self) -> list[str | cv2.Mat]:
        return None
    
    def GetDisplayResults(self) -> list[str | cv2.Mat]:
        return None
    
def HandleProcess(process: ImageProcess):
    while True:
        process.Process()

def GetProcessImageByIndex(processAndThreadList: list[ImageProcess | Thread], imageIndex) -> cv2.Mat:
    currentIndex: int = 0
    for imageProcess in processAndThreadList:
        results = imageProcess[0].GetDisplayResults()
        for textImageSet in results:
            text: str = textImageSet[0]
            image: cv2.Mat = textImageSet[1]
            if currentIndex == imageIndex:
                return image
            currentIndex += 1

def GetProcessTextByIndex(processAndThreadList: list[ImageProcess | Thread], imageIndex) -> str:
    currentIndex: int = 0
    for imageProcess in processAndThreadList:
        results = imageProcess[0].GetDisplayResults()
        for textImageSet in results:
            text: str = textImageSet[0]
            image: cv2.Mat = textImageSet[1]
            if currentIndex == imageIndex:
                return text
            currentIndex += 1

def HandleShowImages(windowName: str, maxWidth: int, maxColumns: int, imageProcessList: List[ImageProcess]):
    assert maxWidth
    assert maxColumns
    assert len(imageProcessList)

    # Create process threads
    processAndThreadList: list[ImageProcess | Thread] = []
    process: ImageProcess
    lastUpdateTimeTime: list[int] = []
    processTreadImageCount: List[int] = []
    for process in imageProcessList:
        thread = Thread(target = HandleProcess, args=[process])
        processAndThreadList.append([process, thread])
        lastUpdateTimeTime.append(0)

    # Initialize values
    windowPrepared: bool = False
    imagePositionList: list[ list[ int | int ] | list[ int | int ] ] = []
    imageCount: int = 0
    imageIndex: int = 0
    imageWidth: int = 0
    imageHeight: int = 0
    columns: int = 0
    rows:int = 0
    fullImageWidth: int = 0
    fullImageHeight: int = 0
    fullImage: cv2.Mat = 0
    upPosition: int = 0
    fontFace = cv2.FONT_HERSHEY_PLAIN
    maxTextCharacters = 20
    thickness = 15
    fontScale: float = 0
    textImageList: List[str | cv2.Mat] = []
    textColor = (255, 255, 255)
    outlineColor = (0, 0, 0)

    # Prepared one time values
    processThread: List[ImageProcess, Thread]
    index: int = 0
    for processThread in processAndThreadList:
        process = processThread[0]
        results = process.GetDisplayResults()
        textImageList.append(results)
        processTreadImageCount.append(len(textImageList))
        imageCount += len(results)

    # Compute the real sizes.
    image = textImageList[0][0][1]
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    columns = imageCount if imageCount < maxColumns else maxColumns
    rows = int(np.ceil(imageCount / columns))
    fullImageWidth = imageWidth * columns
    fullImageHeight = imageHeight * rows
    fullImage = np.zeros((fullImageHeight, fullImageWidth, 3), np.uint8)
    fontScale = GetFontScale(imageWidth, maxTextCharacters, fontFace, thickness)
    upPosition = 0

    # Show images
    imageIndex: int = 0
    processThread: list[ImageProcess | Thread]
    column: int = 0
    leftPosition: int = 0
    for processThread in processAndThreadList:
        process: ImageProcess = processThread[0]
        lineType = cv2.LINE_AA
        outlineDeflection = 10
        for result in process.GetDisplayResults():
            text: str = result[0]
            image: cv2.Mat = result[1]
            imageModified = image.copy()
            textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)

            # Add text to full image
            point = int(imageWidth / 2 - textSize[0] / 2), int(imageWidth - textSize[1] * 1)
            PutOutlinedText(imageModified, text, point, fontFace, fontScale, outlineDeflection, textColor, outlineColor, thickness, lineType)

            imagePosition: list[ list[ int | int ] | list[ int | int ] ] = \
                [upPosition, leftPosition], [upPosition + imageHeight, leftPosition + imageWidth]
            imagePositionList.append(imagePosition)

            ulx: int = imagePosition[0][1]
            uly: int = imagePosition[0][0]
            drx: int = imagePosition[1][1]
            dry: int = imagePosition[1][0]
            assert imageModified is not None
            FillImageRect(fullImage, imageModified, ulx, uly, drx, dry)
            
            column += 1
            leftPosition += imageWidth
            if column >= columns:
                column = 0
                leftPosition = 0
                upPosition += imageHeight

            imageIndex += 1
            if imageIndex >= imageCount:
                break

    imageScale: float = maxWidth / fullImageWidth
    fullImageHeight: float = imageHeight * rows

    scaledImage: cv2.Mat = cv2.resize(fullImage, None, None, imageScale, imageScale)
    scaledImageWidth: int = scaledImage.shape[1]
    scaledImageHeight: int = scaledImage.shape[0]

    def MouseCallback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            imageColumnClicked = int(x / scaledImageWidth * columns)
            imageRowClicked = int(y / scaledImageHeight * rows)
            imageIndex = imageColumnClicked + imageRowClicked * columns
            if imageIndex < imageCount:
                singleImageWindow = "Single"
                image = GetProcessImageByIndex(processAndThreadList, imageIndex)
                cv2.imshow(singleImageWindow, image)
    
    instructionTexts = ["Close window or press any key to exit", "Click on an image to see it fully"]
    showInstructionsDuration = 8000
    currentDuration = 0
    showInstructions = True
    hasShownFinalWindow = False
    instructionImage = None
    
    if showInstructions and showInstructionsDuration > 0 and instructionTexts != None:
        instructionFontThickness = 1
        charactersOnHorizontal = 70
        charactersInBox = 30
        instructionFontScale = GetFontScale(scaledImageWidth, charactersOnHorizontal, fontFace, instructionFontThickness)
        characterWidth, characterHeight = GetFontCharacterSize("Q", fontFace, instructionFontScale, instructionFontThickness)
        point = characterHeight * 2, characterHeight * 3
        textSpacing = int(np.ceil(0.4 * characterHeight))
        newlineHeight = textSpacing + characterHeight
        outlineDeflection = 3
        textColor = 255, 255, 255
        textOutlineColor = 0, 0, 0
        totalTextHeight = int(newlineHeight * len(instructionTexts))
        pointUL = point[0] - int(characterHeight * 1.5), int(point[1] - characterHeight * 1.5)
        pointDR = point[0] + int(charactersInBox * characterWidth), point[1] + totalTextHeight

    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, MouseCallback)

    for [process, thread] in processAndThreadList:
        thread.start()

    runLoop = True
    while runLoop:
        # Check for process's image changes.
        imageIndex: int = 0
        processIndex: int = 0
        for processThread in processAndThreadList:
            process: ImageProcess = processThread[0]
            currentUpdateTime = process.GetLastUpdateTime()
            if currentUpdateTime != lastUpdateTimeTime[processIndex]:
                image: list[str | cv2.Mat]
                displayResults = process.GetDisplayResults()
                for result in displayResults:
                    #print(f"Drawing {imageIndex}")
                    startTime = time.time_ns()
                    text = result[0]
                    image = result[1]
                    updatedImage = image.copy()

                    # Add text to full image
                    point = int(imageWidth / 2 - textSize[0] / 2), int(imageWidth - textSize[1] * 1)
                    textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
                    PutOutlinedText(updatedImage, text, point, fontFace, fontScale, outlineDeflection, textColor, outlineColor, thickness, lineType)

                    imagePosition = imagePositionList[imageIndex]
                    ulx: int = imagePosition[0][1]
                    uly: int = imagePosition[0][0]
                    drx: int = imagePosition[1][1]
                    dry: int = imagePosition[1][0]
                    FillImageRect(fullImage, updatedImage, ulx, uly, drx, dry)

                    scaledImage: cv2.Mat = cv2.resize(fullImage, None, None, imageScale, imageScale)
                    scaledImageWidth: int = scaledImage.shape[1]
                    scaledImageHeight: int = scaledImage.shape[0]                        
                    endTime = time.time_ns()
                    #print(f"Done drawing... time = { TimeNSToS(endTime - startTime)}")
                    imageIndex += 1
                lastUpdateTimeTime[processIndex] = currentUpdateTime
            else:
                imageIndex += len(process.GetDisplayResults())
            processIndex += 1
        cv2.imshow(windowName, scaledImage)
        cv2.waitKey(1)

        if showInstructions == True:
            if currentDuration > showInstructionsDuration:
                showInstructions = False
                instructionImage = np.copy(scaledImage)        
                DrawAlphaRectangle(instructionImage, pointUL, pointDR, 155)
                for i in range(0, len(instructionTexts)):
                    text = instructionTexts[i]
                    PutOutlinedText(instructionImage, text, point, fontFace, fontScale, outlineDeflection, textColor, textOutlineColor, thickness, cv2.LINE_AA)
                    point = point[0], point[1] + newlineHeight
                    cv2.imshow(windowName, instructionImage)
                    cv2.waitKey(1)

        elif hasShownFinalWindow == False:
            hasShownFinalWindow = True
            cv2.imshow(windowName, scaledImage)
            cv2.waitKey(1)
        
        cv2.waitKey(100)
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            runLoop = False
    cv2.destroyAllWindows()

    exit(0)