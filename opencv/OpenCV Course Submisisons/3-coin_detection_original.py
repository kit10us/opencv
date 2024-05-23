# I am learning python as the same time as OpenCV so I will
# have put in more work than I had to. I hope that is okay.
#
# See main at the bottom of the script for the actual assignment
# work.


import sys
from pathlib import Path
import cv2
import numpy as np
import time

def TimeNSToMS(timeNS) -> float:
    return timeNS * 0.001

def TimeMSToS(timeMS) -> float:
    return timeMS * 0.001

def GetImagePath(filename) -> Path:
    localPath: Path = Path(sys.argv[0]).parent / filename
    if localPath.exists():
        return localPath
    
    imageFolderPath: Path = Path(sys.argv[0]).parent.parent / "images" / filename
    if imageFolderPath.exists():
        return imageFolderPath
    
    raise FileNotFoundError(f"Could not locate file {filename}")
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



# This took some time to get right, however, it was well worth it when I could visually see
# each stage of the image processing so I could pick the best solutions. I was even able to
# show side-by-sides of performing the operations with difference values so I could find
# the best numbers.
# All images have to be of the same size.
def ShowImages(windowName: str, maxWidth: int, maxColumns: int, imageList):
    assert maxWidth
    assert maxColumns
    assert len(imageList)

    # Compute the real sizes.
    imageCount: int = len(imageList)
    imageIndex: int = 0
    imageWidth: int = imageList[0][1].shape[1]
    imageHeight: int = imageList[0][1].shape[0]
    columns: int = imageCount if imageCount < maxColumns else maxColumns
    rows:int = int(np.ceil(imageCount / columns))
    fullImageWidth: int = imageWidth * columns
    fullImageHeight: int = imageHeight * rows
    fullImage: cv2.Mat = np.zeros((fullImageHeight, fullImageWidth, 3), np.uint8)
    upPosition: int = 0

    # Compute text values
    fontFace = cv2.FONT_HERSHEY_PLAIN
    maxTextCharacters = 20
    thickness = 15
    fontScale = GetFontScale(imageWidth, maxTextCharacters, fontFace, thickness)

    while imageIndex < imageCount:
        leftPosition: int = 0
        for column in range(0, columns):
            # Add image to full image
            workingImage: cv2.Mat = np.copy(imageList[imageIndex][1])

            # Add text to full image
            text: str = imageList[imageIndex][0]
            textSize, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)

            point = int(imageWidth / 2 - textSize[0] / 2), int(imageWidth - textSize[1] * 1)
            textColor = (255, 255, 255)
            outlineColor = (0, 0, 0)
            lineType = cv2.LINE_AA
            outlineDeflection = 10
            PutOutlinedText(workingImage, text, point, fontFace, fontScale, outlineDeflection, textColor, outlineColor, thickness, lineType)

            fullImage[upPosition:upPosition + imageHeight, leftPosition:leftPosition + imageWidth] = \
                workingImage[:, :]

            imageIndex += 1
            if imageIndex >= imageCount:
                break 
            leftPosition += imageWidth
        upPosition += imageHeight

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
                cv2.imshow(singleImageWindow, imageList[imageIndex][1])
    
    instructionTexts = ["Close window or press any key to exit", "Click on an image to see it fully"]
    showInstructionsDuration = 8000
    currentDuration = 0
    showInstructions = True
    hasShownFinalWindow = False
    instructionImage = None

    
    if showInstructions and showInstructionsDuration > 0 and instructionTexts != None:
        instructionImage = np.copy(scaledImage)        
        fontFace = cv2.FONT_HERSHEY_PLAIN
        maxTextCharacters = 100
        thickness = 1
        charactersOnHorizontal = 70
        charactersInBox = 30
        fontScale = GetFontScale(scaledImageWidth, charactersOnHorizontal, fontFace, thickness)
        characterWidth, characterHeight = GetFontCharacterSize("Q", fontFace, fontScale, thickness)
        point = characterHeight * 2, characterHeight * 3
        textSpacing = int(np.ceil(0.4 * characterHeight))
        newlineHeight = textSpacing + characterHeight
        outlineDeflection = 3
        textColor = 255, 255, 255
        textOutlineColor = 0, 0, 0
        totalTextHeight = int(newlineHeight * len(instructionTexts))
        pointUL = point[0] - int(characterHeight * 1.5), int(point[1] - characterHeight * 1.5)
        pointDR = point[0] + int(charactersInBox * characterWidth), point[1] + totalTextHeight

        DrawAlphaRectangle(instructionImage, pointUL, pointDR, 155)
        for i in range(0, len(instructionTexts)):
            text = instructionTexts[i]
            PutOutlinedText(instructionImage, text, point, fontFace, fontScale, outlineDeflection, textColor, textOutlineColor, thickness, cv2.LINE_AA)
            point = point[0], point[1] + newlineHeight

    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, MouseCallback)
    runLoop = True
    while runLoop:
        if showInstructions == True:
            if currentDuration > showInstructionsDuration:
                showInstructions = False
            cv2.imshow(windowName, instructionImage)
        elif hasShownFinalWindow == False:
            hasShownFinalWindow = True
            cv2.imshow(windowName, scaledImage)
        
        cv2.waitKey(100)
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            runLoop = False
    cv2.destroyAllWindows()
    exit(0)


def PerformContour(sourceImage: cv2.Mat, displayImage: cv2.Mat, allImages):

    # -9-
    # threshold 50, 255
    # erode 5, 5
    # erode 35, 35
    # dilate 47, 47

    workingImage = sourceImage.copy()

    retval, workingImage = cv2.threshold(workingImage, 10, 255, cv2.THRESH_BINARY)
    allImages.append(["Cont. G. Threshold", cv2.cvtColor(workingImage, cv2.COLOR_GRAY2RGB)])

    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    workingImage = cv2.erode(workingImage, erodeKernel)
    allImages.append(["Cont. erode", cv2.cvtColor(workingImage, cv2.COLOR_GRAY2RGB)])

    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    workingImage = cv2.erode(workingImage, erodeKernel)
    allImages.append(["Cont. erode", cv2.cvtColor(workingImage, cv2.COLOR_GRAY2RGB)])

    dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    workingImage = cv2.dilate(workingImage, dilateKernel)
    allImages.append(["Cont. dilate", cv2.cvtColor(workingImage, cv2.COLOR_GRAY2RGB)])


    finalMorphedImage = workingImage
    contours, hierarchy = cv2.findContours(finalMorphedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contoursEDDisplayImage = displayImage.copy()
    cv2.drawContours(contoursEDDisplayImage, contours, -1, (0, 255, 0), 13, cv2.LINE_AA)
    for contour in contours:
        moment = cv2.moments(contour)
        x = int(round(moment["m10"] / moment["m00"]))
        y = int(round(moment["m01"] / moment["m00"]))
        cv2.circle(contoursEDDisplayImage, (x, y), 10, (255, 0, 0), -1)
    allImages.append([f"Final Contours count = {len(contours)}", contoursEDDisplayImage, cv2.COLOR_GRAY2RGB])


def main(argc: int, argv) -> int:
    allImages = []

    # Grade 1 - Read image:
    originalImage = LoadImage(GetImagePath("Coins4.jpg"))
    allImages.append(["Original Image", originalImage])    


    # Grade 2 - Convert image to grayscale:
    grayscaleImageName = "Grayscale"
    grayscaleImage = cv2.cvtColor(originalImage, cv2.COLOR_RGB2GRAY)
    allImages.append(["Gray Image", cv2.cvtColor(grayscaleImage, cv2.COLOR_GRAY2RGB)])


    # Grade 3 - Split image into R,G,B channels:
    blueChannelImage = originalImage[:, :, 0]
    greenChannelImage = originalImage[:, :, 1]
    redChannelImage = originalImage[:, :, 2]
    allImages.append(["Blue Channel Image", cv2.cvtColor(blueChannelImage, cv2.COLOR_GRAY2RGB)])
    allImages.append(["Green Channel Image", cv2.cvtColor(greenChannelImage, cv2.COLOR_GRAY2RGB)])
    allImages.append(["RedChannel Image", cv2.cvtColor(redChannelImage, cv2.COLOR_GRAY2RGB)])


    # Grade 4 - Perform thresholding
    # I played with a number of difference threshold values, I also tried various bitwise operations. 
    # I found that the green channel worked best with the following numbers.
    retval, thresholdImageBlue = cv2.threshold(blueChannelImage, 70, 255, cv2.THRESH_BINARY)
    retval, thresholdImageGreen = cv2.threshold(greenChannelImage, 70, 255, cv2.THRESH_BINARY)
    retval, thresholdImageRed = cv2.threshold(redChannelImage, 70, 255, cv2.THRESH_BINARY)
    allImages.append(["Threshold Blue Image", cv2.cvtColor(thresholdImageBlue, cv2.COLOR_GRAY2RGB)])
    allImages.append(["Threshold Green Image", cv2.cvtColor(thresholdImageGreen, cv2.COLOR_GRAY2RGB)])
    allImages.append(["Threshold Red Image", cv2.cvtColor(thresholdImageRed, cv2.COLOR_GRAY2RGB)])


    # Grade 5 - Perform morphological operations
    # I tried to get good results from erode, then dilate, but it wasn't quite right.
    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    erodedImage = cv2.erode(thresholdImageGreen, erodeKernel)
    dilateImage = cv2.dilate(thresholdImageGreen, dilateKernel)
    allImages.append(["Eroded Image", cv2.cvtColor(erodedImage, cv2.COLOR_GRAY2RGB)])
    allImages.append(["Dilated Image", cv2.cvtColor(dilateImage, cv2.COLOR_GRAY2RGB)])

    # So I found that doing an erode, then dilate. I tried another erodem, which looked better,
    # it just made the blob detection detect too many blobs within the coins themselves. So
    # I backed it off. Yes, I used a high number for the kernel. With more time and need
    # I'd create a tool to play with these settings.
    # Reduce Noice
    morphedImage = cv2.erode(thresholdImageGreen, erodeKernel)
    morphedImage = cv2.dilate(morphedImage, dilateKernel)
    allImages.append(["Morphed Image", cv2.cvtColor(morphedImage, cv2.COLOR_GRAY2RGB)])


    # Grade 6 - Detect coins (blob detection)
    # I had a lot of problems here, but I read more on what the different values mean, so
    # I kept tweaking. I might make some tools later to play with all of these features
    # functionality.
    sbdParams = cv2.SimpleBlobDetector.Params()
    sbdParams.blobColor = 255
    sbdParams.minDistBetweenBlobs = 2
    sbdParams.filterByArea = False
    sbdParams.filterByCircularity = True
    sbdParams.minCircularity = 0.8
    sbdParams.filterByConvexity = True
    sbdParams.minCircularity = 0.1
    sbdParams.filterByInertia = True
    sbdParams.minInertiaRatio = 0.2
    sbDetector = cv2.SimpleBlobDetector.create(sbdParams)
    keypoints = sbDetector.detect(morphedImage)


    # 7 - Display the detected coins
    # Pretty straight forward here, except that there is not enough consitency between the
    # order of vertex positions. So I had to reverse them here (this order, x then y, is what
    # I am used to).
    blobsImage = originalImage.copy()
    radius = 20
    centerColor = (0, 200, 0)
    circumferenceColor = (200, 0, 0)
    thickness = 20
    for key in keypoints:
        point = int(key.pt[0]), int(key.pt[1])
        cv2.circle(blobsImage, (point[0], point[1]), radius, centerColor, thickness)
        cv2.circle(blobsImage, (point[0], point[1]), int(key.size / 2), circumferenceColor, thickness)
    allImages.append(["Blob Detection Image ({})".format(len(keypoints)), blobsImage])

    
    # Grade 8 - Perform CCA
    # Not as pretty as yours. When I have more experience with the values of all of these
    # features, they'll be pretty.        
    _, imageLabels = cv2.connectedComponents(morphedImage)
    minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(imageLabels)
    imageLabels = 255 * (imageLabels - minValue) / (maxValue - minValue)
    imageLabels = np.uint8(imageLabels)
    colorMappedImage = cv2.applyColorMap(imageLabels, cv2.COLORMAP_JET)
    allImages.append(["Color Mapped Image", colorMappedImage])


    # Grade 9 = Perform contour detection and fit circles
    # Found this the most difficult, however, once I added a smaller erode before
    # the next larger erode and dilate, it got the results I wanted.
    PerformContour(greenChannelImage, originalImage, allImages)

    # As mentioned with the ShowImages function, this helped me see the results faster
    # and thus iterate through a lot of changes quickly.
    # I hadn't noticed my image was inverse (white coins), however, I liked it better so
    # stuck with it.  
    ShowImages("Window", 1200, 5, allImages)

    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))