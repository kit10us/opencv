import sys
from pathlib import Path
import cv2
import numpy as np
from enum import Enum
import time

windowName = "Window"
totalPlayTime = 10000 # In seconds

def GetImagePath(filename) -> Path:
    return Path(sys.argv[0]).parent / filename

def GetOutputPath(filename) -> Path:
    return Path(sys.argv[0]).parent / "output" / filename

def CreateSampleImage() -> cv2.Mat:
    im = np.zeros  ((10,10),dtype='uint8')
    im[0,1]     = 1
    im[-1,0]    = 1
    im[-2,-1]   = 1
    im[2,2]     = 1
    im[5:8,5:8] = 1
    return im

class MorphType(Enum):
    DILATE = 0
    ERODE = 1

def DrawWindow(window: str, image: cv2.Mat, text: str):
    imageHeight, imageWidth = image.shape[0:2]
    textFontFace = cv2.FONT_HERSHEY_PLAIN
    textScale = 1
    textColor = (255, 255, 255)
    textThickness = 1
    textLineType = cv2.LINE_AA
    textSize = cv2.getTextSize(text, textFontFace, textScale, textThickness)[0]
    textPosition = (int(imageWidth / 2 - textSize[0] / 2), int(imageHeight - 2 * textSize[1]))
    drawWindow = image[:,:]
    drawWindow = cv2.putText(drawWindow, text, (textPosition[0] + 2, textPosition[1] + 2), textFontFace, textScale, (0, 0, 0), textThickness, textLineType)
    drawWindow = cv2.putText(drawWindow, text, textPosition, textFontFace, textScale, textColor, textThickness, textLineType)
    cv2.imshow(window, drawWindow)


def Morph(source: cv2.Mat, kernel: cv2.Mat, morph: MorphType, captureFilename: str = None) -> cv2.Mat:
    global windowName
    global totalPlayTime

    scaleDimension = 512 / (source.shape[0] if source.shape[0] > source.shape[1] else source.shape[1])
    finalImageSize = (int(scaleDimension * source.shape[1]), int(scaleDimension * source.shape[0]))
    msPerPixel = int(totalPlayTime / source.shape[0] / source.shape[1])
    captureFPS = int(1000 / msPerPixel)
    shapeHeight, shapeWidth = kernel.shape[0], kernel.shape[1]
    sourceHeight, sourceWidth = source.shape[0], source.shape[1]

    videoWriter = None
    if captureFilename is not None:
        videoWriter = cv2.VideoWriter(captureFilename, cv2.VideoWriter.fourcc(*'avc1'), captureFPS, finalImageSize)

    borderTop = shapeHeight // 2
    borderBottom = shapeHeight - borderTop - shapeHeight % 2
    borderLeft = shapeWidth // 2
    borderRight = shapeWidth - borderLeft - shapeWidth % 2

    borderImage = np.zeros((sourceHeight + borderTop + borderBottom, sourceWidth + borderTop + borderBottom))
    borderImage = cv2.copyMakeBorder(source, borderTop, borderBottom, borderLeft, borderRight, cv2.BORDER_CONSTANT, value = 0)
    if morph == MorphType.ERODE:
        kernel = cv2.bitwise_not(kernel)

    for y in range(0, sourceHeight):
        for x in range(0, sourceWidth):
            if morph == MorphType.DILATE:
                if source[y, x] == 1:
                    borderArea = (y, y + shapeHeight, x, x + shapeWidth)
                    borderImage[borderArea[0]:borderArea[1], borderArea[2]:borderArea[3]] = \
                        cv2.bitwise_or(borderImage[borderArea[0]:borderArea[1], borderArea[2]:borderArea[3]], kernel)
            elif morph == MorphType.ERODE:
                if source[y, x] == 0:
                    borderArea = (y, y + shapeHeight, x, x + shapeWidth)
                    borderImage[borderArea[0]:borderArea[1], borderArea[2]:borderArea[3]] = \
                        cv2.bitwise_and(borderImage[borderArea[0]:borderArea[1], borderArea[2]:borderArea[3]], kernel)

            finalImage = borderImage * 255
            finalImage[y + 1:y + 2, x + 1: x + 2] = 128
            finalImage = cv2.resize(finalImage[1:-1, 1:-1], finalImageSize, interpolation = cv2.INTER_NEAREST)
            videoWriter.write(finalImage)

            DrawWindow(windowName, finalImage, morph.name)
            cv2.waitKey(msPerPixel)
            if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
                if videoWriter is not None:
                    videoWriter.release()
                exit(0)

    if videoWriter is not None:
        videoWriter.release()
    return borderImage[borderTop:borderTop + sourceHeight, borderLeft:borderLeft + sourceWidth]

def main(argc: int, argv) -> int:
    cv2.namedWindow(windowName)
    cv2.moveWindow(windowName, 100, 100)

    sourceImage = CreateSampleImage()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    erodeVideoName = GetOutputPath("erode.mp4").as_posix()
    finalImage = Morph(sourceImage, kernel, MorphType.ERODE, erodeVideoName)
    scaleDimension = finalImage.shape[0] if finalImage.shape[0] > finalImage.shape[1] else finalImage.shape[1]
    finalImageScale = 512 / scaleDimension
    finalImage *= 255
    finalImage = cv2.resize(finalImage, None, None, finalImageScale, finalImageScale, cv2.INTER_NEAREST)
    DrawWindow(windowName, finalImage, "Erode done")
    cv2.waitKey(4000)
    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        exit(0)

    finalImage = cv2.erode(sourceImage, kernel)
    scaleDimension = finalImage.shape[0] if finalImage.shape[0] > finalImage.shape[1] else finalImage.shape[1]
    finalImageScale = 512 / scaleDimension
    finalImage *= 255
    finalImage = cv2.resize(finalImage, None, None, finalImageScale, finalImageScale, cv2.INTER_NEAREST)
    DrawWindow(windowName, finalImage, "Erode expected")
    cv2.waitKey(4000)
    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        exit(0)

    dilateVideoName = GetOutputPath("dilate.mp4").as_posix()
    finalImage = Morph(sourceImage, kernel, MorphType.DILATE, dilateVideoName)
    scaleDimension = finalImage.shape[0] if finalImage.shape[0] > finalImage.shape[1] else finalImage.shape[1]
    finalImageScale = 512 / scaleDimension
    finalImage *= 255
    finalImage = cv2.resize(finalImage, None, None, finalImageScale, finalImageScale, cv2.INTER_NEAREST)
    DrawWindow(windowName, finalImage, "Dilate done")
    cv2.waitKey(4000)

    finalImage = cv2.dilate(sourceImage, kernel)
    scaleDimension = finalImage.shape[0] if finalImage.shape[0] > finalImage.shape[1] else finalImage.shape[1]
    finalImageScale = 512 / scaleDimension
    finalImage *= 255
    finalImage = cv2.resize(finalImage, None, None, finalImageScale, finalImageScale, cv2.INTER_NEAREST)
    DrawWindow(windowName, finalImage, "Dilate expected")
    cv2.waitKey(4000)
    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        exit(0)

    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))